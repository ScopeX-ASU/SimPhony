from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.registry import MODELS
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn.modules.utils import _pair
from torch.types import Device, _size

from .quantized_base_layer import QBaseLayer
from .utils import ActQuantizer_LSQ, WeightQuantizer_LSQ

__all__ = [
    "QConv2d",
]


@MODELS.register_module()
class QConv2d(QBaseLayer):
    """
    Quantized Conv2d layer.
    """

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    _in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size,
        stride: _size = 1,
        padding: _size = 0,
        dilation: _size = 1,
        groups: int = 1,
        bias: bool = False,
        w_bit: int = 8,
        in_bit: int = 8,
        out_bit: int = 8,
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        # assert (
        #     groups == 1
        # ), f"Currently group convolution is not supported, but got group: {groups}"

        self.w_bit = w_bit
        self.in_bit = in_bit
        self.out_bit = out_bit

        self.build_parameters()

        ## quantization tool
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

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def build_parameters(self):
        weight = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            bias=False,
        ).weight.data
        self.weight = Parameter(weight)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _forward_impl(self, x: Tensor, weight: Tensor) -> Tensor:
        x = F.conv2d(
            x,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return x

    def extra_repr(self):
        s = (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}"
            f", stride={self.stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += f", padding={self.padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias:
            s += f", bias={self.bias}"
        s += f", in_bits={self.in_bit}, w_bits={self.w_bit}, out_bits={self.out_bit}"
        s += f", input_noise_std={self.input_noise_std}, weight_noise_std={self.weight_noise_std}, output_noise_std={self.output_noise_std}"
        return s
