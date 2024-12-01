import torch
from mmengine.registry import MODELS
from pyutils.compute import add_gaussian_noise
from torch import Tensor, nn
from torch.types import Device

from .utils import (
    ActQuantizer_LSQ,
    WeightQuantizer_LSQ,
)

__all__ = [
    "QMatMul",
]


@MODELS.register_module()
class QMatMul(nn.Module):
    """
    Customized MatMul module to replace a torch function.
    """

    def __init__(
        self,
        w_bit: int = 8,
        in_bit: int = 8,
        out_bit: int = 8,
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        super().__init__()
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.out_bit = out_bit
        self.device = device
        # self.phase_noise_std = 1e-5
        ### build trainable parameters
        ### quantization tool
        self.input_quantizer = ActQuantizer_LSQ(
            None,
            device=self.device,
            nbits=self.in_bit,
            offset=True,
            signed=False,
            mode="tensor_wise",
        )
        self.weight_quantizer = WeightQuantizer_LSQ(
            None,
            device=self.device,
            nbits=self.w_bit,
            offset=False,
            signed=True,
            mode="tensor_wise",
        )
        self.output_quantizer = ActQuantizer_LSQ(
            None,
            device=self.device,
            nbits=self.out_bit,
            offset=True,
            signed=False,
            mode="tensor_wise",
        )

        self.set_input_noise(0)
        self.set_weight_noise(0)
        self.set_output_noise(0)

        self.set_input_bitwidth(self.in_bit)
        self.set_weight_bitwidth(self.w_bit)
        self.set_output_bitwidth(self.out_bit)

    def set_input_noise(self, noise_std: float = 0.0) -> None:
        self.input_noise_std = noise_std

    def set_weight_noise(self, noise_std: float = 0.0) -> None:
        self.weight_noise_std = noise_std

    def set_output_noise(self, noise_std: float = 0.0) -> None:
        self.output_noise_std = noise_std

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bit(in_bit)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.weight_quantizer.set_bit(w_bit)

    def set_output_bitwidth(self, out_bit: int) -> None:
        self.out_bit = out_bit
        self.output_quantizer.set_bit(out_bit)

    def weight_transform(self, weight: Tensor) -> Tensor:
        if self.w_bit < 16:
            weight = self.weight_quantizer(weight)  # [-alpha, alpha]
            alpha = self.weight_quantizer.alpha.data
        else:
            alpha = weight.data.abs().max()
        self.weight_scale = alpha.item()

        if self.weight_noise_std > 0:
            ## Warning: noise need to be added to normalized input
            weight = add_gaussian_noise(
                weight, noise_std=self.weight_noise_std * self.weight_scale
            )
        return weight

    def input_transform(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)  # [-alpha, alpha] or [0, alpha]
            alpha = self.input_quantizer.alpha.data
        else:
            alpha = x.data.abs().max()
        self.input_scale = alpha.item()

        if self.input_noise_std > 0:
            x = add_gaussian_noise(x, noise_std=self.input_noise_std * self.input_scale)

        return x

    def output_transform(self, x: Tensor) -> Tensor:
        if self.output_noise_std > 0:
            in_dim = x.size(-1)  # Number of columns
            x = add_gaussian_noise(
                x,
                noise_std=self.output_noise_std
                * self.input_scale
                * self.weight_scale
                * (in_dim**0.5),
            )

        if self.out_bit < 16:
            x = self.output_quantizer(x)

        return x

    def forward(self, x: Tensor, weight: Tensor) -> Tensor:
        x = self.input_transform(x)
        weight = self.weight_transform(weight)
        y = torch.matmul(x, weight)
        # print("Yes,. I am here")
        y = self.output_transform(y)
        return y

    def extra_repr(self):
        s = "QuantizedMatMul"

        s += f", in_bits={self.in_bit}, w_bits={self.w_bit}, out_bits={self.out_bit}"
        s += f", input_noise_std={self.input_noise_std}, weight_noise_std={self.weight_noise_std}, output_noise_std={self.output_noise_std}"
        return s
