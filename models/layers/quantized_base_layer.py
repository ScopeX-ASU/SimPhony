from typing import Dict

import torch
from pyutils.compute import add_gaussian_noise
from torch import Tensor, nn

from torch.types import Device


__all__ = ["QBaseLayer"]


class QBaseLayer(nn.Module):
    def __init__(self, *args, device: Device = torch.device("cpu"), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # cuda or cpu, defaults to cpu
        self.device = device

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

    # def _add_output_noise(self, x) -> None:
    #     if self.output_noise_std > 1e-6:
    #         # vector_len = np.prod(self.weight.shape[1::2])  # q*c*k2
    #         noise = gen_gaussian_noise(
    #             x,
    #             noise_mean=0,
    #             noise_std=np.sqrt(vector_len) * self.output_noise_std,
    #         )

    #         x = x + noise
    #     return x
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
            if hasattr(self, "in_features"):
                in_dim = self.in_features
            elif hasattr(self, "in_channels"):
                in_dim = self.in_channels
            else:
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

    def forward(self, x: Tensor) -> Tensor:
        # try:
        # ## preprocess input
        #     x = self.input_transform(x)
        # except:
        #     import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        x = self.input_transform(x)
        ## preprocess weights
        weight = self.weight_transform(self.weight)

        # import pdb; pdb.set_trace()
        ## forward pass
        out = self._forward_impl(x, weight)

        ## postprocess output
        out = self.output_transform(out)

        ## add bias
        if self.bias is not None:
            out = out + self.bias

        return out

    def _forward_impl(self, x: Tensor, weights: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""
