import inspect
from typing import Dict, Optional, Union

import torch
from mmcv.cnn.bricks import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.registry import MODELS
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn  # , set_deterministic
from torch.types import Device, _size

# from torchonn.op.mrr_op import *

__all__ = [
    "LinearBlock",
    "ConvBlock",
    "MatMulBlock",
    "QBaseModel",
]


def build_linear_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    if cfg is None:
        cfg_ = dict(type="Linear")
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if inspect.isclass(layer_type):
        return layer_type(*args, **kwargs, **cfg_)  # type: ignore
    # Switch registry to the target scope. If `linear_layer` cannot be found
    # in the registry, fallback to search `linear_layer` in the
    # mmengine.MODELS.
    with MODELS.switch_scope_and_registry(None) as registry:
        linear_layer = registry.get(layer_type)
    if linear_layer is None:
        raise KeyError(
            f"Cannot find {linear_layer} in registry under scope "
            f"name {registry.scope}"
        )
    layer = linear_layer(*args, **kwargs, **cfg_)

    return layer


def build_matmul_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    if cfg is None:
        cfg_ = dict(type="MatMul")
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if inspect.isclass(layer_type):
        return layer_type(*args, **kwargs, **cfg_)  # type: ignore
    # Switch registry to the target scope. If `linear_layer` cannot be found
    # in the registry, fallback to search `linear_layer` in the
    # mmengine.MODELS.
    with MODELS.switch_scope_and_registry(None) as registry:
        matmul_layer = registry.get(layer_type)
    if matmul_layer is None:
        raise KeyError(
            f"Cannot find {matmul_layer} in registry under scope "
            f"name {registry.scope}"
        )
    layer = matmul_layer(*args, **kwargs, **cfg_)

    return layer


class MatMulBlock(nn.Module):
    def __init__(
        self,
        matmul_cfg: dict = dict(type="QMatMul"),
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout, inplace=False) if dropout > 0 else None
        if matmul_cfg["type"] not in {"QMatMul", None}:
            matmul_cfg.update({"device": device})
        self.matmul = build_matmul_layer(
            matmul_cfg,
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z = self.matmul(x, y)
        return z


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        linear_cfg: dict = dict(type="QLinear"),
        norm_cfg: dict | None = None,
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
        dropout: float = 0.0,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False) if dropout > 0 else None
        if linear_cfg["type"] not in {"Linear", None}:
            linear_cfg.update({"device": device})
        self.linear = build_linear_layer(
            linear_cfg,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_features)
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        dilation: Union[int, _size] = 1,
        groups: int = 1,
        bias: bool = False,
        conv_cfg: dict = dict(type="QConv2d"),
        norm_cfg: dict | None = dict(type="BN", affine=True),
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ) -> None:
        super().__init__()
        conv_cfg = conv_cfg.copy()
        if conv_cfg["type"] not in {"Conv2d", None}:
            conv_cfg.update({"device": device})
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if norm_cfg is not None:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class QBaseModel(nn.Module):
    def __init__(
        self,
        *args,
        conv_cfg=dict(type="QConv2d"),
        linear_cfg=dict(type="QLinear"),
        matmul_cfg=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        with MODELS.switch_scope_and_registry(None) as registry:
            self._conv = (registry.get(conv_cfg["type"]),)
            self._linear = (registry.get(linear_cfg["type"]),)
            self._matmul = (registry.get(matmul_cfg["type"]),) if matmul_cfg else None
            self._conv_linear = self._conv + self._linear
            self._conv_linear += self._matmul if self._matmul else ()

    def reset_parameters(self, random_state: int = None) -> None:
        for name, m in self.named_modules():
            if isinstance(m, self._conv_linear):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_output_noise(self, noise_std: float = 0.0):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_output_noise"
            ):
                layer.set_output_noise(noise_std)

    def set_input_noise(self, noise_std: float = 0.0) -> None:
        self.input_noise_std = noise_std
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_input_noise"
            ):
                layer.set_input_noise(noise_std)

    def set_weight_noise(self, noise_std: float = 0.0) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_weight_noise"
            ):
                layer.set_weight_noise(noise_std)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_weight_bitwidth"
            ):
                layer.set_weight_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_input_bitwidth"
            ):
                layer.set_input_bitwidth(in_bit)

    def set_output_bitwidth(self, out_bit: int) -> None:
        self.out_bit = out_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_output_bitwidth"
            ):
                layer.set_input_bitwidth(out_bit)

    def build_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "build_parameters"
            ):
                layer.build_parameters()
