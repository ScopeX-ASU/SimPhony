import os
import sys
from typing import Callable, List, Tuple

import einops
import numpy as np
import pandas as pd
import torch
from pyutils.compute import gen_gaussian_filter2d
from pyutils.general import logger
from pyutils.quant.lsq import get_default_kwargs_q, grad_scale, round_pass
from scipy.interpolate import LinearNDInterpolator
from torch import Tensor, nn
from torch.nn import Parameter
from torch.types import _size

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))


__all__ = [
    "STE",
    "mzi_out_diff_to_phase",
    "mzi_phase_to_out_diff",
    "PhaseVariationScheduler",
    "CrosstalkScheduler",
    "merge_chunks",
    "partition_chunks",
    "ActQuantizer_LSQ",
    "WeightQuantizer_LSQ",
]

DEBUG = False



def polynomial(x: Tensor, coeff: Tensor) -> Tensor:
    ## coeff: from high to low order coefficient, last one is constant
    ## e.g., [p5, p4, p3, p2, p1, p0] -> p5*x^5 + p4*x^4 + p3*x^3 + p2*x^2 + p1*x + p0
    # print(x.shape)
    out = 0
    for i in range(coeff.size(0) - 1, 0, -1):
        out = out + x.pow(i).mul_(coeff[coeff.size(0) - i - 1])
    out.add_(coeff[-1])
    return out


def polynomial2(
    x: Tensor | float, y: Tensor | float, coeff: Tensor | List[float]
) -> Tensor | float:
    ## coeff [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]
    if len(coeff) == 3:
        # coeff [1, x, y]
        return coeff[0] + coeff[1] * x + coeff[2] * y
    elif len(coeff) == 6:
        # coeff [1, x, y, x^2, xy, y^2]
        return (
            coeff[0]
            + coeff[1] * x
            + coeff[2] * y
            + coeff[3] * x**2
            + coeff[4] * x * y
            + coeff[5] * y**2
        )
    elif len(coeff) == 10:
        # coeff [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]
        x_2, y_2 = x**2, y**2
        return (
            coeff[0]
            + coeff[1] * x
            + coeff[2] * y
            + coeff[3] * x_2
            + coeff[4] * y * x
            + coeff[5] * y_2
            + coeff[6] * x_2 * x
            + coeff[7] * y * x_2
            + coeff[8] * y_2 * x
            + coeff[9] * y_2 * y
        )
    else:
        raise NotImplementedError


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_noisy):
        ## use x_noisy as forward
        return x_noisy

    @staticmethod
    def backward(ctx, grad_output):
        ## gradient flow back to x, not x_noisy
        return grad_output.clone(), None


def mzi_out_diff_to_phase(x: Tensor) -> Tensor:
    """Y-branch-based 1x2 MZI.
    The power difference on two output ports, i.e., out_diff = out1 - out2, converted to the internal arm phase difference (delta_phi)
    delta_phi \in [-pi/2, pi/2], if delta_phi > 0, heat up upper arm phase shifter; if delta_phi < 0, heat up lower arm phase shifter
    out_diff \in [-1, 1] ideally with infinite extinction ratio.
    out1 = 0.5(1-sin(delta_phi))
    out2 = 0.5(1+sin(delta_phi))
    out_diff = out1-out2=-sin(delta_phi)
    delta_phi = arcsin(out_diff), need to make sure delta_phi is in the range of [-pi/2, pi/2]
    phase shifter: exp(-jdelta_phi), delta_phi is phase lag.

    Args:
        x (Tensor): output port power difference of the 1x2 MZI

    Returns:
        Tensor: delta phi
    """
    return torch.asin(
        -x.clamp(-1, 1)
    )  # this clamp is for safety, as the input x may not be exactly in [-1, 1]


def mzi_phase_to_out_diff(x: Tensor) -> Tensor:
    """Y-branch-based 1x2 MZI.
    The internal arm phase difference (delta_phi) converted to the power difference on two output ports, i.e., out_diff = out1 - out2
    delta_phi \in [-pi/2, pi/2], if delta_phi > 0, heat up upper arm phase shifter; if delta_phi < 0, heat up lower arm phase shifter
    out_diff \in [-1, 1] ideally with infinite extinction ratio.
    out1 = 0.5(1-sin(delta_phi))
    out2 = 0.5(1+sin(delta_phi))
    out_diff = out1-out2=-sin(delta_phi)

    Args:
        x (Tensor): delta phi

    Returns:
        Tensor: output port power difference of the 1x2 MZI
    """
    return -torch.sin(x)


class PhaseVariationScheduler(object):
    def __init__(
        self,
        size: _size = [
            4,
            4,
            8,
            8,
        ],  # this one should be the architecture dimension, [R, C, K, K], not workload dimension [P, Q, K, K]
        T_max: int = 1000,  # total number of steps
        mean_schedule_fn: Callable = lambda: 0.02,  # a function that returns a mean value for a given step
        std_schedule_fn: Callable = lambda: 0.01,  # a function that returns a std value for a given step
        smoothing_kernel_size: int = 5,  # kernel size for the gaussian filter
        smoothing_factor: float = 0.05,  # how smooth is the distribution
        smoothing_mode: str = "core",  # smoothing mode, core: smooth core-by-core, arch: smooth over all cores
        min_std: float = 0.001,
        momentum: float = 0.9,  # how much is on previous noise std distribution, momenutm * old_map + (1-momentum) * new_map
        noise_scenario_src: str = "",  # corner, edge
        noise_scenario_tgt: str = "",
        random_state: int = 0,
        device="cuda:0",
    ) -> None:
        """
        Each device has a zero-mean random phase noise, the phase noise follows N(0, std_i^2) for the i-th device
        Then we need a tensor `noise_std_map` with the same shape as `phase` for each device.
        The noise intensity for each device will gradually drift to an unknown direction.
        To create a random std drift curve, e.g., std_i=0.01 -> std_i=0.008 -> std_i=0.012 -> std_i=0.011 -> std_i=0.009
        we construct a random process, std_i = momentum * std_i_old + (1 - momentum) * std_i_new
        , where std_i_new is randomly sampled from a Gaussian distribution N(std_mean_i, std_std_i),
        std_i_new are spatially smooth across all devices, therefore we apply gaussian filter to smooth `noise_std_map`.
        std_mean_i is controlled by mean_schedule_fn, std_std_i is controlled by std_schedule_fn.
        For example, if std_mean increases, it means the average noise intensity increases across all devices. Maybe the environment gets worse or background noises become larger.
        For example, if std_std increases, it means the noise intensity becomes more diverse across all devices. Maybe there is some local perturbation that makes devices behave diversely.

        """
        # std of the phase noise follows Gaussian distribution ~ N(noise_std_mean, noise_std_std^2)
        super().__init__()
        self.size = size
        self.T_max = T_max
        self.mean_schedule_fn = mean_schedule_fn
        self.std_schedule_fn = std_schedule_fn
        self.smoothing_kernel_size = smoothing_kernel_size
        assert (
            smoothing_kernel_size == 0 or smoothing_kernel_size % 2 == 1
        ), "Must have 0 or odd size of kernel"
        self.smoothing_factor = smoothing_factor
        self.smoothing_mode = smoothing_mode
        self.momentum = momentum
        self.min_std = min_std
        self.noise_scenario_src = noise_scenario_src
        self.noise_scenario_tgt = noise_scenario_tgt

        self.random_state = random_state
        self.device = device
        self.core_noise_mean_map = None

        if self.smoothing_factor > 0 and self.smoothing_kernel_size > 0:
            self.gaussian_filter = gen_gaussian_filter2d(
                self.smoothing_kernel_size,
                std=self.smoothing_factor,
                center_one=False,
                device=self.device,
            )[None, None, ...].to(device)
            # print(self.gaussian_filter)
            # exit(0)
            pad = self.smoothing_kernel_size // 2
            self.padder = torch.nn.ReflectionPad2d((pad, pad, pad, pad))
        else:
            self.gaussian_filter = None
        self.noises = None

        self.reset()

    def reset(self):
        self._step = 0
        self.noise_std_mean = self.mean_schedule_fn(
            0
        )  # the mean of the phase noise std
        self.noise_std_std = self.std_schedule_fn(0)  # the std of the phase noise std
        self.noise_std_map = None
        self.noises = None
        self.noise_scenario_transition()
        self.update_noise_std_map()

    def step(self):
        # one time step to change noise distribution
        # you can call this at any frequency you want, e.g., every step, or even more fine-grained (every layer)
        self._step += 1  # enable periodic scheduling
        self.noise_std_mean = self.mean_schedule_fn(
            (self._step % self.T_max) / self.T_max  # 0.002 # enable periodic scheduling
        )  # normalized value to query the mean schedule function
        self.noise_std_std = self.std_schedule_fn(
            (self._step % self.T_max) / self.T_max  # enable periodic scheduling
        )  # normalized value to query the std schedule function
        self.update_noise_std_map()
        self.noise_scenario_transition()
        # print(f'noise_std_mean={self.noise_std_mean}, noise_std_std={self.noise_std_std}')

    def noise_scenario_transition(self):
        if self.noise_scenario_tgt == "edge":
            target_core_noise_mean_map = torch.tensor(
                [
                    [0.008, 0.006, 0.004, 0.002],
                    [0.008, 0.006, 0.004, 0.002],
                    [0.008, 0.006, 0.004, 0.002],
                    [0.008, 0.006, 0.004, 0.002],
                ],
                device=self.device,
            )
        elif self.noise_scenario_tgt == "corner":
            target_core_noise_mean_map = torch.tensor(
                [
                    [0.008, 0.006, 0.006, 0.004],
                    [0.006, 0.006, 0.004, 0.004],
                    [0.006, 0.004, 0.004, 0.002],
                    [0.004, 0.004, 0.002, 0.002],
                ],
                device=self.device,
            )

        core_noise_mean_map = self._generate_core_noise_mean_map()
        if self.core_noise_mean_map is None:
            self.core_noise_mean_map = core_noise_mean_map
        else:
            self.core_noise_mean_map = (
                self.momentum * self.core_noise_mean_map
                + (1 - self.momentum) * target_core_noise_mean_map
            )

    def _generate_core_noise_mean_map(self) -> Tensor:
        core_noise_mean_map = torch.zeros(self.size[:-2])
        if self.noise_scenario_src == "corner":
            self.core_noise_mean_map = torch.tensor(
                [
                    [0.0008, 0.0006, 0.0006, 0.0004],
                    [0.0006, 0.0006, 0.0004, 0.0004],
                    [0.0006, 0.0004, 0.0004, 0.0002],
                    [0.0004, 0.0004, 0.0002, 0.0002],
                ]
            )
        elif self.noise_scenario_src == "edge":
            self.core_noise_mean_map = torch.tensor(
                [
                    [0.0008, 0.0006, 0.0004, 0.0002],
                    [0.0008, 0.0006, 0.0004, 0.0002],
                    [0.0008, 0.0006, 0.0004, 0.0002],
                    [0.0008, 0.0006, 0.0004, 0.0002],
                ]
            )
        else:
            raise NotImplementedError

        core_noise_mean_map = self.core_noise_mean_map / 2
        self.core_noise_mean_map = core_noise_mean_map.to(self.device)
        return core_noise_mean_map.to(self.device)

    def _generate_noise_std_map(self):
        # determinstic, do not affect external random state, different across steps
        # this is device-wise noise std map. Each MRR has a noise_std, representing its noise intensity
        # this can create difference within each core for intra-core remapping
        noise_std_map = torch.normal(
            self.noise_std_mean,
            self.noise_std_std,
            size=self.size,
            generator=torch.Generator(device=self.device).manual_seed(
                self.random_state + self._step
            ),
            device=self.device,
        ).clamp_min_(
            self.min_std
        )  # [c, r, k, k] # the std needs to be at least some small value, if std is zero, then it is not random at all.
        ## we assume each core has a background noise intensity(std) specific to this core.
        ## this core-wise std will be added to the noise_std_map. then different cores can have different noise intensity
        ## this core-wise noise intensity leads to unbalanced/uneven noise levels across c x r cores. Then enable inter-core remapping.
        # self._generate_core_noise_mean_map()

        core_noise_std_map = (
            torch.normal(
                mean=self.core_noise_mean_map,  # self.noise_std_mean, #core_noise_mean_map, # core-wise std level
                std=self.noise_std_std,  # core-wise std diversity
                # size=self.size[:-2], # std_mean for this core, approximated by the std_mean averaged across kxk rings
                generator=torch.Generator(device=self.device).manual_seed(
                    self.random_state + self._step
                ),
                # device=self.device,
            )
            .clamp_min_(self.min_std)
            .to(self.device)[..., None, None]
        )  # [c,r,1,1]  # the std needs to be at least some small value, if std is zero, then it is not random at all.

        # print(core_noise_std_map.shape)
        # print(noise_std_map.shape)
        # =========================================================  core-wise noise_mean_map  ========================================================#
        ## core-wise noise_mean_map, different core has different noise intensity, and we define 2 modes for noise_mean distribution
        ## 1: corner mode, noise intensity is most significant at left-up core, and smoothly distributed along x- and y- axis
        ## 2: edge mode, noise intensity is most significant at left column and distributed along x-axis

        noise_std_map = (core_noise_std_map + noise_std_map) / 2
        if self.gaussian_filter is not None:
            # we assume the noise intensity (i.e., std) distribution is smooth locally
            if self.smoothing_mode == "core":
                noise_std_map = torch.nn.functional.conv2d(
                    self.padder(noise_std_map).flatten(0, 1).unsqueeze(1),
                    self.gaussian_filter,
                    padding="valid",
                ).view_as(noise_std_map)
            elif self.smoothing_mode == "arch":
                noise_std_map = partition_chunks(
                    torch.nn.functional.conv2d(
                        self.padder(merge_chunks(noise_std_map)[None, None]),
                        self.gaussian_filter,
                        padding="valid",
                    )[0, 0],
                    bs=noise_std_map.shape[-1],
                )
        return noise_std_map

    def update_noise_std_map(self):
        noise_std_map = self._generate_noise_std_map()
        if self.noise_std_map is None:
            self.noise_std_map = noise_std_map
        else:
            # every time step, we gradually update the noise std map to another random map, the momentum controls how much we keep the old map
            self.noise_std_map = (
                self.momentum * self.noise_std_map + (1 - self.momentum) * noise_std_map
            )

    def sample_noise(self, size=None):
        ## size [P, Q, k, k]: the workload size you want to map to this [R, C, K, K] multi-core MRR accelerator
        ## If size is None, then the workload is assumed to be [R, C, K, K]
        ## need to return [P, Q, k, k] phase noises for this workload
        ## assume the archiecture is [R, C, k, k]

        # when size=self.size, i.e., batch = [1, 1], then P=R, Q=C, i.e., each block in the layer weight matrix is mapped to a photonic core.
        # when batch = [u, v], we assume u=\ceil{P/R}, v=\ceil{Q/C}, i.e., the matrix needs to be partition into multiple RkxCk blocks and mapped sequentially to the same accelerator.
        size = size or self.size
        batch = (
            int(np.ceil(size[0] / self.size[0])),
            int(np.ceil(size[1] / self.size[1])),
        )

        # we assume the phase noise has zero mean, only std is determined by the noise_std_map
        # the P, Q, K, K workload will be chunked into u-by-v chunks (with same padding), each chunk is R, C, K, K, and thus can be mapping to the arch.
        # The u-by-v chunks require u-by-v times inference. The u-by-v inferences will see the same noise distribution, but different noise samples.
        # noise_std_map = einops.repeat(self.noise_std_map, "r c k l-> (u r) (v c) k l", u=batch[0], v=batch[1])[:size[0], :size[1]]
        noise_std_map = einops.repeat(
            self.noise_std_map, "r c k l-> u v r c k l", u=batch[0], v=batch[1]
        )

        noise_std_map = (
            noise_std_map.permute(0, 2, 1, 3, 4, 5)
            .flatten(0, 1)
            .flatten(1, 2)[: size[0], : size[1]]
        )

        noises = torch.normal(
            mean=0.0, std=noise_std_map
        )  # n ~ N(0, noise_std_map^2) different device has different std
        # noises = torch.normal(
        #     mean=0.0, std=self.noise_std_map
        # )  # n ~ N(0, noise_std_map^2) different device has different std
        self.noises = noises  ## add this to record the noise sampled.
        return noises


class MZIPowerEvaluator(object):
    def __init__(
        self,
        csv_file: str = "./MZIdata/MZIPower.csv",
        interv_s: int = 10,
        ps_width: int = 6,
        device="cuda:0",
    ):
        self.ps_width = ps_width
        self.device = device
        self.set_spacing(interv_s)
        assert os.path.exists(csv_file), f"{csv_file} does not exist"
        self._fit_MZI_power_interp(csv_file)

    def set_spacing(self, interv_s: int):
        self.interv_s = interv_s

    def _fit_MZI_power_interp(self, csv_file: str = "./MZIdata/MZIPower.csv"):
        df = pd.read_csv(csv_file)
        # end = 22
        end = -5  # 7um at P_pi
        power = df.iloc[2:end, 1].astype("float32").to_numpy()
        power = np.concatenate([np.array([0]), power])
        delta_phases = []
        distances = []
        for i in range(1, 13):
            delta_phi = df.iloc[2:end, 1 + 4 * i].astype("float32").to_numpy() - (
                df.iloc[2:end, 4 * i].astype("float32").to_numpy()
            )
            # when delta_phi=0, power must be 0
            delta_phi = np.concatenate([np.array([0]), delta_phi])
            distance = (
                df.iloc[0:1, i * 4 - 2]
                .astype("float32")
                .repeat(len(delta_phi))
                .to_numpy()
            )
            delta_phases.append(delta_phi)
            distances.append(distance)
        # print(power)
        # print(delta_phases)
        ideal_distance = distances.pop(-1)  # noqa: F841
        ideal_delta_phases = delta_phases.pop(-1)
        # ideal_distance = distances[-1]
        # ideal_delta_phases = delta_phases[-1]
        ## delta phase cannot be higher then ideal case
        ## larger distance must have larger delta phase
        for i in range(1, len(delta_phases)):
            delta_phases[i] = np.minimum(
                np.maximum(delta_phases[i - 1], delta_phases[i]), ideal_delta_phases
            )
        X_data = np.stack(
            [np.concatenate(delta_phases, 0), np.concatenate(distances, 0)], -1
        )
        # print(X_data)
        Y_data = np.concatenate([power] * len(delta_phases), 0)
        # print(Y_data)

        self._MZI_power_interp = LinearNDInterpolator(X_data, Y_data)
        return self._MZI_power_interp

    def calc_MZI_power(
        self,
        delta_phi: Tensor | float,
        interv_s: float | None = None,
        reduction: str = "sum",
    ) -> Tensor:
        ## delta_phi: phase difference of an MZI, input must be -pi/2 to pi/2
        interv_s = interv_s or self.interv_s
        if not torch.is_tensor(delta_phi):
            is_tensor = False
            delta_phi = torch.tensor(delta_phi, device=self.device)
        else:
            is_tensor = True
        interv_s = max(self.ps_width + 1, min(interv_s, 25))
        if self._MZI_power_interp is None:
            self._fit_MZI_power_interp()
        delta_phi_shape = delta_phi.shape
        delta_phi = delta_phi.flatten()
        X_data = (
            torch.stack(
                [delta_phi.abs(), torch.ones_like(delta_phi).fill_(interv_s)], -1
            )
            .cpu()
            .numpy()
        )
        power = (
            torch.from_numpy(self._MZI_power_interp(X_data).reshape(delta_phi_shape))
            .float()
            .to(self.device)
        )  # [p,q,r,c,k1,k2]
        # print(power)
        # exit(0)
        # power = polynomial2(
        #     delta_phi.abs(),
        #     interv_s,
        #     self.power_coefficients,
        # ).relu() # nonnegative power
        if reduction == "sum":
            power = power.sum()
        elif reduction == "mean":
            power = power.mean()
        if not is_tensor:
            power = power.item()
        return power


class CrosstalkScheduler(object):
    def __init__(
        self,
        crosstalk_coupling_factor: Tuple[float, ...] = [
            3.55117528e-07,
            -1.55789201e-05,
            -8.29631681e-06,
            9.89616761e-03,
            -1.76013871e-01,
            1,
        ],  # y=p1*x^5+p2*x^4+p3*x^3+p4*x^2+p5*x+p6
        crosstalk_exp_coupling_factor: float = [
            0.2167267,
            -0.12747211,
        ],  # a * exp(b*x)
        interv_h: float = 24.0,  # horizontal spacing (unit: um) between the center of two MZIs
        interv_v: float = 120.0,  # vertical spacing (unit: um) between the center of two MZIs
        interv_s: float = 10.0,  # horizontal spacing (unit: um) between two arms of an MZI
        ps_width: float = 6,  # phase shifter width (unit: um)
        power_coefficients: tuple[float, ...] = [
            1.2659822,
            5.5226021e00,
            -2.8833720e-01,
            3.8191843e-01,
            -2.4061684e-01,
            1.9616380e-02,
            -3.5375733e-02,
            -5.9136748e-04,
            5.7989508e-03,
            -4.0769577e-04,
        ],  # [1, a, b, a^2, ab, b^2, a^3, a^2b, ab^2, b^3]
        device="cuda:0",
    ) -> None:
        super().__init__()
        self.crosstalk_coupling_factor = torch.tensor(
            crosstalk_coupling_factor, device=device
        )
        self.ps_width = ps_width
        self.crosstalk_exp_coupling_factor = crosstalk_exp_coupling_factor
        self.mzi_power_evaluator = MZIPowerEvaluator(
            interv_s=interv_s, ps_width=ps_width, device=device
        )
        self.set_spacing(interv_h, interv_v, interv_s)
        self.device = device
        self.crosstalk_matrix = None
        self.power_coefficients = torch.tensor(power_coefficients, device=device)

    def set_spacing(
        self,
        interv_h: None | float = None,
        interv_v: None | float = None,
        interv_s: None | float = None,
    ) -> None:
        self.interv_h = interv_h or self.interv_h
        self.interv_v = interv_v or self.interv_v
        self.interv_s = interv_s or self.interv_s
        self.mzi_power_evaluator.set_spacing(self.interv_s)
        assert (
            self.interv_h >= self.interv_s + self.ps_width
        ), f"Horizontal spacing ({self.interv_h}) should be larger than the width of an MZI ({self.interv_s}+{self.ps_width}={self.interv_s+self.ps_width})"

    def get_crosstalk_matrix(self, phase) -> Tensor:
        """Generate the crosstalk coupling matrix Gamma given the array size
        Assume the layout of the array as: accumultion along one column

        Args:
            size (Tuple[int,int]): (..., dim_y, dim_x), i.e., (..., #cols, #rows)

        Returns:
            Gamma (Tensor): crosstalk coupling matrix of size (..., size_y*size_x, size_y*size_x)
            ## g1,1,     g1,2,     g1,3,     g_1,k1*k2
            ## g2,1,     g2,2,     g2,3,     g_2,k1*k2
            ## ...
            ## gk1*k2,1, gk1*k2,2, gk1*k2,3, g_k1*k2,k1*k2
            ## need to left multiply by this Gamma, i.e., phase' = Gamma x phase or phase' = phase x Gamma^T
            gamma_i,j is defined as |delta_phi(j)| causes an gamma_i,j * |delta_phi(j)| increase to delta_phi(i)
            i.e., delta_phi(i)' = delta_phi(i) + gamma_i,j * |delta_phi(j)|
            gamma_i,j can be positive if upper arm of j-th MZI is closer to the i-th MZI
            gamma_i,j can be negative if lower arm of j-th MZI is closer to the i-th MZI
            ## gamma = exp(-factor * distance_upper) - exp(-factor * distance_lower), crosstalk follows exponential delay
            ## since the gamma is dynamically dependent on the phase, each k1 x k2 block has its own gamma matrix
        """
        return self._get_crosstalk_matrix(
            self.crosstalk_coupling_factor,
            phase,
            self.interv_h,
            self.interv_v,
            self.interv_s,
        )

    def _get_crosstalk_gamma(
        self, distance: Tensor, crosstalk_coupling_factor: Tensor
    ) -> Tensor:
        ## less than 23 um, polynomial is more accurate, higher than 20 um, exp is more accurate
        gamma = torch.where(
            distance < 23,
            polynomial(distance, crosstalk_coupling_factor),
            self.crosstalk_exp_coupling_factor[0]
            * torch.exp(self.crosstalk_exp_coupling_factor[1] * distance),
        )
        if DEBUG:
            mask = (distance < 23) & (distance > 10)
            distance = distance[mask]
            gammav = gamma[mask]
            gammav, indices = gammav.sort(descending=True)
            distance = distance[indices]
            print(self.interv_h)
            # print(distance)
            print(distance[:6])
            print(gammav[:6])
        # print(crosstalk_coupling_factor)
        # print(polynomial(torch.tensor([8,9,10,11,12,19,20,23.], device=mask.device), crosstalk_coupling_factor))
        return gamma

    def _get_crosstalk_matrix(
        self, crosstalk_coupling_factor, phase, interv_h, interv_v, interv_s
    ) -> Tensor:
        k1, k2 = phase.shape[-2:]  # k1=#cols, k2=#rows
        X, Y = torch.meshgrid(
            torch.arange(k1, device=self.device, dtype=torch.float),
            torch.arange(k2, device=self.device, dtype=torch.float),
        )
        X, Y = X.flatten(), Y.flatten()

        mask = phase.data.flatten(-2, -1).unsqueeze(-2) < 0
        X_distance = X.unsqueeze(0).sub(X.unsqueeze(1)).mul_(interv_h)  # [k1*k2, k1*k2]
        Y_distance_sq = Y.unsqueeze(0).sub(Y.unsqueeze(1)).square_().mul_(interv_v**2)
        distance_upper = (
            X_distance.sub(interv_s * mask)  # [..., k1*k2,k1*k2]
            .square_()
            .add_(Y_distance_sq)
        ).sqrt_()

        distance_lower = (
            X_distance.add(interv_s * (~mask))  # [..., k1*k2,k1*k2]
            .square_()
            .add_(Y_distance_sq)
        ).sqrt_()

        self.crosstalk_matrix = self._get_crosstalk_gamma(
            distance_upper, crosstalk_coupling_factor
        ).sub_(self._get_crosstalk_gamma(distance_lower, crosstalk_coupling_factor))
        self.crosstalk_matrix[..., torch.arange(k1 * k2), torch.arange(k1 * k2)] = 1.0

        return self.crosstalk_matrix

    def apply_crosstalk(self, phase: Tensor, crosstalk_matrix: Tensor) -> Tensor:
        """Apply thermal crosstalk to phases.
        need to left multiply by this crosstalk_matrix (Gamma), i.e., phase' = Gamma x phase or phase' = phase x Gamma^T
        phases are from [-pi/2, pi/2].
        If phase > 0, heat up upper arm.
        If phase < 0, heat up lower arm.
        No matter phases are postive or negative, it is heated up with higher temperature. So its contribution is related to |phases|.
        e.g., if a phase is -pi/2, it contributes 1% delta_phi increase to another MZI. the delta_phi of victim MZI will increase by 1%*|-pi/2|=pi/200.
        The current function can handle pruned phases. Once pruned, delta_phi=0, the pruned phase has zero impact to other MZIs, but itself will still be
        impacted by other MZIs.
        Args:
            phase (Tensor): [..., k1, k2] blocked phases
            crosstalk_matrix (Tensor): [..., k1*k2, k1*k2] crosstalk matrix Gamma

        Returns:
            Tensor: _description_
        """
        phase_shape = phase.shape
        flat_phase = phase.flatten(-2, -1)
        ## we want to create
        """
         phi1  |phi1| |phi1|
        |phi2|  phi2  |phi2|
        |phi3| |phi3|  phi3
        """
        flat_phase_abs = flat_phase.abs()[..., None].expand(
            [-1] * flat_phase.dim() + [flat_phase.shape[-1]]
        )

        flat_phase_abs = torch.diagonal_scatter(flat_phase_abs, flat_phase, 0, -2, -1)

        ## then we need batched vector inner product
        return torch.einsum("...ij,...ji->...i", crosstalk_matrix, flat_phase_abs).view(
            phase_shape
        )

    def calc_crosstalk_score(self, mask: Tensor, is_col: bool = True) -> float:
        ## given a sparsity mask, find its crosstalk score, higher score means less crosstalk
        ## the crosstalk is the sum of the negative exp distance between active elements, sum(exp(-d_ij))
        active_indices = torch.nonzero(mask).squeeze()  # e.g., [0, 1, 3, 5]
        num_active = active_indices.numel()
        if num_active < 2:
            # Not enough active elements to calc separation or density.
            # should be higher than the best case with two actives
            ## treated as 2 actives with distance of k
            active_indices = torch.tensor([0, mask.shape[0]], device=mask.device)
            num_active = 2

        # calc distances between consecutive active elements

        node_spacing = self.interv_v if is_col else self.interv_h

        ## MZI center to center distance, can be positive or negative
        ## positive means aggressor is to the right of the victim (left/right layout)
        index_distance = (
            active_indices.unsqueeze(0)
            .sub(active_indices.unsqueeze(1))  # [k1*k2, k1*k2]
            .float()
        )
        index_distance[index_distance.abs() > 1] = (
            200  # only differentiate with 1 MZI in between, otherwise, set a large distance.
        )
        center_dist = index_distance.mul_(node_spacing)
        ## to estimate, aggressor MZI's center to victim's upper arm, can be positive or negative

        distance_upper = (center_dist - self.interv_s / 2).abs()

        ## to estimate, aggressor MZI's center to victim's lower arm, can be positive or negative
        distance_lower = (center_dist + self.interv_s / 2).abs()

        ## because we do not know the actual aggressor's phase, we do not know whether to include interv_s or not.
        ## We use a rough estimation, here we overestimate the crosstalk.
        total_crosstalk = self._get_crosstalk_gamma(
            distance_upper, self.crosstalk_coupling_factor
        ).sub_(
            self._get_crosstalk_gamma(distance_lower, self.crosstalk_coupling_factor)
        )
        total_crosstalk[torch.arange(num_active), torch.arange(num_active)] = 0.0

        ## column-wise sum, sum over all aggressors for each victim, positive/negative might cancel out
        ## then accumulate the absolute crosstalk factors over all victims
        ## '-' to make the score higher the better
        return -total_crosstalk.sum(1).abs().sum().item()

    def calc_crosstalk_scores(self, masks: Tensor, is_col: bool = True) -> Tensor:
        ## given a sparsity mask, find its crosstalk score, higher score means less crosstalk
        ## the crosstalk is the sum of the negative exp distance between active elements, sum(exp(-d_ij))
        ## mask: [num_combinations, array_length]
        ## return: [num_combinations]
        shape = masks.shape[:-1]
        total_crosstalk = [
            self.calc_crosstalk_score(m, is_col) for m in masks.flatten(0, -2)
        ]
        return torch.tensor(total_crosstalk, device=self.device).view(shape)

    def calc_MZI_power(
        self, delta_phi: Tensor, interv_s: float = None, reduction: str = "sum"
    ) -> Tensor:
        power = self.mzi_power_evaluator.calc_MZI_power(delta_phi, interv_s, reduction)
        return power


def merge_chunks(x: Tensor) -> Tensor:
    # x = [h1, w1, h2, w2, ...., hk, wk]
    # out: [h1*h2*...*hk, w1*w2*...*wk]

    dim = x.dim()
    x = x.permute(
        list(range(0, dim, 2)) + list(range(1, dim + 1, 2))
    )  # x = [h, bs, w, bs]
    x = x.reshape(np.prod([x.shape[i] for i in range(dim // 2)]), -1)

    return x


def partition_chunks(x: Tensor, out_shape: int | Tuple[int, ...]) -> Tensor:
    ### x: [h1*h2*...*hk, w1*w2*...*wk]
    ### out_shape: (h1, w1, ...)
    ### out: [h1, w1, h2, w2, ...., hk, wk]
    in_shape = list(out_shape[::2]) + list(out_shape[1::2])
    x = x.reshape(in_shape)  # [h1, h2, ..., hk, w1, w2, ..., wk]
    x = x.permute(
        torch.arange(len(out_shape)).view(2, -1).t().flatten().tolist()
    )  # [h1, w1, h2, w2, ...., hk, wk]
    return x


class ActQuantizer_LSQ(nn.Module):
    def __init__(self, in_features: int, device="cuda:0", **kwargs_q):
        super().__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q)
        self.nbits = kwargs_q["nbits"]
        if self.nbits <= 0:  # no need to enable quantize
            self.register_parameter("alpha", None)
            return
        self.q_mode = kwargs_q["mode"]
        self.offset = kwargs_q["offset"]
        self.zero_point = None
        self.device = device
        if self.q_mode == "kernel_wise":
            self.alpha = Parameter(torch.empty(in_features, device=device))
            if self.offset:
                self.zero_point = Parameter(torch.empty(in_features, device=device))
                torch.nn.init.zeros_(self.zero_point)
        else:
            self.alpha = Parameter(torch.empty(1, device=device))
            if self.offset:
                self.zero_point = Parameter(torch.tensor([0.0], device=device))

        self.register_buffer("init_state", torch.zeros(1))
        self.register_buffer("signed", torch.zeros(1))

    def update_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q["nbits"] = self.nbits = nbits
        self._compute_quant_range()

    def _compute_quant_range(self):
        if self.signed == 1:
            self.Qn = -(2 ** (self.nbits - 1))
            self.Qp = 2 ** (self.nbits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2**self.nbits - 1

    def extra_repr(self):
        if self.alpha is None:
            return "fake"
        return "{}".format(self.kwargs_q)

    def _initialize_state(self, x):
        logger.info(
            f"LSQ Act quantizer: (mode: {self.q_mode}): initialize weight scale for int{self.nbits} quantization with offset: {self.offset}"
        )
        if self.q_mode == "kernel_wise":
            logger.info(f"Scale dimension: {self.alpha.shape}")
        # choose implementation from https://github.com/YanjingLi0202/Q-ViT/blob/main/Quant.py
        # if x.numel() == 0:
        #     # 处理空张量情况
        #     self._compute_quant_range()
        #     self.init_state.fill_(1)
        #     return

        # if x.dim() == 0:
        #     min_val = x.data
        # elif x.dim() == 1:
        #     min_val = x.data.min()
        # else:
        #     min_val = x.flatten().min()

        if (
            x.data.min() < -1e-5
        ):  # there are significant negative values we will use signed representation
            self.signed.data.fill_(1)
        self._compute_quant_range()
        self.alpha.data.copy_(x.data.abs().mean().mul_(2 / self.Qp**0.5))
        if self.offset:
            self.zero_point.data.copy_(
                self.zero_point.data * 0.9
                + 0.1 * (x.data.min() - self.alpha.data * self.Qn)
            )
        self.init_state.fill_(1)

    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            self._initialize_state(x)

        assert self.init_state == 1

        # g = 1.0 / (max(x.data.numel() * self.Qp, 1e-10)) ** 0.5
        g = 1.0 / (x.data.numel() * self.Qp) ** 0.5

        self.alpha.data.clamp_(min=1e-4)

        alpha = grad_scale(self.alpha, g)  # scale alpha's gradient by g

        if len(x.shape) == 2:  # linear layer
            alpha = alpha.unsqueeze(0)
        elif len(x.shape) == 3:  # attension linear layer
            alpha = alpha.unsqueeze(0).unsqueeze(2)
        elif len(x.shape) == 4:  # conv layer
            # import pdb; pdb.set_trace()
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            raise NotImplementedError

        if self.offset:
            zero_point = (
                self.zero_point.round() - self.zero_point
            ).detach() + self.zero_point
            zero_point = grad_scale(zero_point, g)
            if len(x.shape) == 2:
                zero_point.unsqueeze(0)
            elif len(x.shape) == 3:
                zero_point.unsqueeze(0).unsqueeze(2)
            elif len(x.shape) == 4:
                zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            else:
                raise NotImplementedError
            # zero_point = (
            #     if len(x.shape) == 2
            #         zero_point.unsqueeze(0)
            #     if len(x.shape) == 2
            #     else zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            # )
            # if len(x.shape) == 3:
            #     import pdb; pdb.set_trace()
            x = round_pass((x / alpha + zero_point).clamp(self.Qn, self.Qp))
            x = (x - zero_point) * alpha
        else:
            # if len(x.shape) == 3:
            #     import pdb; pdb.set_trace()
            x = round_pass((x / alpha).clamp(self.Qn, self.Qp)).mul(alpha)

        return x


class WeightQuantizer_LSQ(nn.Module):
    def __init__(self, out_features: int, device="cuda:0", **kwargs_q):
        super().__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q)
        self.nbits = kwargs_q["nbits"]
        if self.nbits <= 0:  # no need to enable quantize
            self.register_parameter("alpha", None)
            return
        self.q_mode = kwargs_q["mode"]
        self.offset = kwargs_q["offset"]
        self.zero_point = None
        self.device = device
        if self.q_mode == "kernel_wise":
            self.alpha = Parameter(torch.empty(out_features, device=device))
            if self.offset:
                self.zero_point = Parameter(torch.empty(out_features, device=device))
                torch.nn.init.zeros_(self.zero_point)
        else:
            self.alpha = Parameter(torch.empty(1, device=device))
            if self.offset:
                self.zero_point = Parameter(torch.tensor([0.0], device=device))

        self.register_buffer("init_state", torch.zeros(1))
        self.register_buffer("signed", torch.tensor([kwargs_q["signed"]]))

    def update_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q["nbits"] = nbits
        self._compute_quant_range()

    def _compute_quant_range(self):
        if self.signed == 1:
            self.Qn = -(2 ** (self.nbits - 1))
            self.Qp = 2 ** (self.nbits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2**self.nbits - 1

    def extra_repr(self):
        if self.alpha is None:
            return "fake"
        return "{}".format(self.kwargs_q)

    def _initialize_state(self, x):
        logger.info(
            f"LSQ Weight quantizer: (mode: {self.q_mode}): initialize weight scale for int{self.nbits} quantization with offset: {self.offset}"
        )
        if self.q_mode == "kernel_wise":
            logger.info(f"Scale dimension: {self.alpha.shape}")

        # if x.numel() == 0:
        #     # 处理空张量情况
        #     self._compute_quant_range()
        #     self.init_state.fill_(1)
        #     return

        self._compute_quant_range()
        self.alpha.data.copy_(x.data.abs().mean().mul_(2 / self.Qp**0.5))
        if self.offset:
            self.zero_point.data.copy_(
                self.zero_point.data * 0.9
                + 0.1 * (x.data.min() - self.alpha.data * self.Qn)
            )
        self.init_state.fill_(1)

    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            self._initialize_state(x)

        assert self.init_state == 1

        g = 1.0 / (x.data.numel() * self.Qp) ** 0.5

        self.alpha.data.clamp_(min=1e-4)

        alpha = grad_scale(self.alpha, g)  # scale alpha's gradient by g

        if len(x.shape) == 2:  # linear layer
            alpha = alpha[..., None]
        elif len(x.shape) == 3:  # attention layer
            alpha = alpha[..., None, None]
        elif len(x.shape) == 4:  # conv layer
            alpha = alpha[..., None, None, None]
        elif len(x.shape) == 6:
            alpha = alpha[..., None, None, None, None, None]
        else:
            raise NotImplementedError

        if self.offset:
            zero_point = round_pass(self.zero_point)
            zero_point = grad_scale(zero_point, g)
            if len(x.shape) == 2:
                zero_point = zero_point[..., None]
            elif len(x.shape) == 3:
                zero_point = zero_point[..., None, None]
            elif len(x.shape) == 4:
                zero_point = zero_point[..., None, None, None]
            elif len(x.shape) == 6:
                zero_point = zero_point[..., None, None, None, None, None]
            x = round_pass((x / alpha + zero_point).clamp(self.Qn, self.Qp))
            x = (x - zero_point) * alpha
        else:
            x = round_pass((x / alpha).clamp(self.Qn, self.Qp)).mul(alpha)

        return x
