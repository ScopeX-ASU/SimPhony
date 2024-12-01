from typing import List, Tuple, Union

import torch
from torch import Tensor, nn
from torch.types import Device, _size

from models.layers.utils import *

from .quantized_base import ConvBlock, LinearBlock, MatMulBlock, QBaseModel

__all__ = ["QMobileViT", "InvertedResidual", "QTransformerEncoderLayer"]


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expand_ratio,
        conv_cfg=dict(type="QConv2d"),
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU6", inplace=True),
        device=torch.device("cuda"),
    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Pointwise convolution (expansion)
            layers.append(
                conv1x1(
                    in_planes=inp,
                    out_planes=hidden_dim,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    device=device,
                )
            )
        # Depthwise convolution
        layers.append(
            conv3x3(
                in_planes=hidden_dim,
                out_planes=hidden_dim,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                device=device,
            )
        )
        # Pointwise convolution (projection)
        layers.append(
            conv1x1(
                in_planes=hidden_dim,
                out_planes=oup,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,  # No activation
                device=device,
            )
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        if self.use_res_connect:
            out = x + self.conv(x)
            # import pdb; pdb.set_trace()
            return x + self.conv(x)
        else:
            out = self.conv(x)
            # import pdb; pdb.set_trace()
            return self.conv(x)


class QAttention(nn.Module):
    """
    Quantized Attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        matmul_cfg=dict(type="QMatMul"),
        linear_cfg=dict(type="QLinear"),
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.inner_dim = dim_head * num_heads
        self.project_out = not (num_heads == 1 and dim_head == dim)
        # assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

        self.qkv = Linear(
            in_features=self.dim,
            out_features=self.inner_dim * 3,
            bias=False,
            device=device,
            linear_cfg=linear_cfg,
            dropout=0.0,
        )

        self.proj = (
            Linear(
                in_features=self.inner_dim,
                out_features=self.dim,
                bias=False,
                device=device,
                linear_cfg=linear_cfg,
                dropout=0.0,
            )
            if self.project_out
            else nn.Identity()
        )

        self.quantized_matmul = MatMul(
            device=device,
            matmul_cfg=matmul_cfg,
        )

    def forward(self, x: Tensor) -> Tensor:
        # import pdb; pdb.set_trace()
        B, N, C = x.shape  # Batch size, sequence length, embedding dimension (dim)
        # Project input to QKV
        qkv = self.qkv(x)  # Shape: [B, N, inner_dim * 3]
        # import pdb; pdb.set_trace()
        qkv = qkv.reshape(
            B, N, 3, self.num_heads, self.dim_head
        )  # [B, N, 3, num_heads, dim_head]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, dim_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, num_heads, N, dim_head]
        # Scaled dot-product attention
        attn_scores = (
            self.quantized_matmul(q, k.transpose(-2, -1)) * self.scale
        )  # [B, num_heads, N, N]
        attn_probs = torch.softmax(attn_scores, dim=-1)  # [B, num_heads, N, N]

        # Apply attention weights to values
        attn_output = self.quantized_matmul(
            attn_probs, v
        )  # [B, num_heads, N, dim_head]

        # Concatenate heads and project back to dim
        attn_output = (
            attn_output.transpose(1, 2).contiguous().reshape(B, N, self.inner_dim)
        )  # [B, N, inner_dim]

        output = self.proj(attn_output)  # [B, N, dim]
        
        return output

    def extra_repr(self):
        s = "Quantized_Attention"
        return s


class QTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        nhead,
        dim_heads,
        mlp_dim,
        dropout=0.0,
        linear_cfg=dict(type="QLinear"),
        matmul_cfg=dict(type="QMatMul"),
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        super(QTransformerEncoderLayer, self).__init__()
        self.self_attn = QAttention(
            dim,
            nhead,
            dim_head=dim_heads,
            linear_cfg=linear_cfg,
            matmul_cfg=matmul_cfg,
            device=device,
        )
        self.linear1 = Linear(
            dim,
            mlp_dim,
            linear_cfg=linear_cfg,
            device=device,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(
            mlp_dim,
            dim,
            linear_cfg=linear_cfg,
            device=device,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class QMobileViTBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        channels,
        patch_size=(2, 2),
        mlp_dim=2048,
        dropout=0.0,
        conv_cfg=dict(type="QConv2d"),
        linear_cfg=dict(type="QLinear"),
        matmul_cfg=dict(type="QMatMul"),
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU6", inplace=True),
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ) -> None:
        super(QMobileViTBlock, self).__init__()
        self.ph, self.pw = patch_size
        self.conv1 = conv3x3(
            channels,
            channels,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            device=device,
        )
        self.conv2 = conv1x1(
            channels,
            dim,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            device=device,
        )
        self.transformer_layers = nn.ModuleList(
            [
                QTransformerEncoderLayer(
                    dim=dim,
                    nhead=4,
                    dim_heads=8,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    linear_cfg=linear_cfg,
                    matmul_cfg=matmul_cfg,
                    device=device,
                )
                for _ in range(depth)
            ]
        )
        self.conv3 = conv1x1(
            dim,
            channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            device=device,
        )
        self.conv4 = conv3x3(
            2 * channels,
            channels,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            device=device,
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        # Unfold into patches
        B, C, H, W = y.shape
        y = y.permute(0, 2, 3, 1).contiguous()
        y = y.view(B, H * W, C)

        for layer in self.transformer_layers:
            y = layer(y)

        # Reshape back to image
        y = y.view(B, H, W, C)
        y = y.permute(0, 3, 1, 2).contiguous()

        y = self.conv3(y)
        y = torch.cat((x, y), dim=1)
        y = self.conv4(y)
        return y


class QMobileViT(QBaseModel):
    def __init__(
        self,
        dim: List[int] = [144, 192, 240],
        depth: List[int] = [2, 4, 3],
        channels: List[int] = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
        patch_size: Tuple[int, int] = (2, 2),
        image_height: int = 256,
        image_width: int = 256,
        num_classes: int = 1000,
        expansion: int = 4,
        conv_cfg: dict = dict(type="QConv2d"),
        linear_cfg: dict = dict(type="QLinear"),
        matmul_cfg: dict = dict(type="QMatMul"),
        norm_cfg: dict = dict(type="BN"),
        act_cfg: dict = dict(type="ReLU6", inplace=True),
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        super().__init__(
            conv_cfg=conv_cfg,
            linear_cfg=linear_cfg,
            matmul_cfg=matmul_cfg,
            # device=device
        )

        assert (
            image_height % patch_size[0] == 0 and image_width % patch_size[1] == 0
        ), "Image dimensions must be divisible by the patch size"

        modules = []
        # Stage 1
        self.conv1 = conv3x3(
            3,
            channels[0],
            stride=2,
            conv_cfg=conv_cfg,
            device=device,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        # Stage 2
        modules.append(
            InvertedResidual(
                inp=channels[0],
                oup=channels[1],
                stride=1,
                expand_ratio=expansion,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                device=device,
            )
        )

        # Stage 3
        modules.append(
            InvertedResidual(
                inp=channels[1],
                oup=channels[2],
                stride=2,
                expand_ratio=expansion,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                device=device,
            )
        )

        modules.append(
            InvertedResidual(
                inp=channels[2],
                oup=channels[3],
                stride=1,
                expand_ratio=expansion,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                device=device,
            )
        )

        # Stage 4
        modules.append(
            InvertedResidual(
                inp=channels[3],
                oup=channels[4],
                stride=2,  # Downsample once
                expand_ratio=expansion,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                device=device,
            )
        )
        # Stage 5
        modules.append(
            QMobileViTBlock(
                dim=dim[0],
                depth=depth[0],
                channels=channels[5],
                patch_size=patch_size,
                mlp_dim=int(dim[0] * 2),
                device=device,
                conv_cfg=conv_cfg,
                linear_cfg=linear_cfg,
                matmul_cfg=matmul_cfg,
            )
        )

        # Stage 6
        modules.append(
            InvertedResidual(
                inp=channels[5],
                oup=channels[6],
                stride=2,  # Downsample once
                expand_ratio=expansion,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                device=device,
            )
        )

        # Stage 7
        modules.append(
            QMobileViTBlock(
                dim=dim[1],
                depth=depth[1],
                channels=channels[7],
                patch_size=patch_size,
                mlp_dim=int(dim[1] * 4),
                device=device,
                conv_cfg=conv_cfg,
                linear_cfg=linear_cfg,
                matmul_cfg=matmul_cfg,
            )
        )

        # Stage 8
        modules.append(
            InvertedResidual(
                inp=channels[7],
                oup=channels[8],
                stride=2,  # Downsample once
                expand_ratio=expansion,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                device=device,
            )
        )

        # Stage 79
        modules.append(
            QMobileViTBlock(
                dim=dim[2],
                depth=depth[2],
                channels=channels[9],
                patch_size=patch_size,
                mlp_dim=int(dim[2] * 4),
                device=device,
                conv_cfg=conv_cfg,
                linear_cfg=linear_cfg,
                matmul_cfg=matmul_cfg,
            )
        )
        self.mobvit_blocks = nn.Sequential(*modules)
        # Stage 9
        self.conv_last = conv1x1(
            channels[-2],
            channels[-1],
            conv_cfg=conv_cfg,
            device=device,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.final_layer = Linear(
            in_features=channels[-1],
            out_features=num_classes,
            bias=False,
            device=device,
            linear_cfg=linear_cfg,
            norm_cfg=None,
            act_cfg=None,
        )

    def forward(self, x):
        x = self.conv1(x)

        x = self.mobvit_blocks(x)

        x = self.conv_last(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.final_layer(x)
        # exit(0)
        return x


def conv3x3(
    in_planes,
    out_planes,
    bias: bool = False,
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
    device: Device = torch.device("cuda"),
    conv_cfg: dict = dict(type="QConv2d"),
    groups: int = 1,
    act_cfg=None,
    norm_cfg=None,
):
    conv = ConvBlock(
        in_planes,
        out_planes,
        3,
        bias=bias,
        stride=stride,
        padding=padding,
        groups=groups,
        conv_cfg=conv_cfg,
        act_cfg=act_cfg,
        norm_cfg=norm_cfg,
        device=device,
    )

    return conv


def conv1x1(
    in_planes,
    out_planes,
    bias: bool = False,
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
    device: Device = torch.device("cuda"),
    conv_cfg: dict = dict(type="QConv2d"),
    act_cfg=None,
    norm_cfg=None,
):
    conv = ConvBlock(
        in_planes,
        out_planes,
        1,
        bias=bias,
        stride=stride,
        padding=padding,
        conv_cfg=conv_cfg,
        act_cfg=act_cfg,
        norm_cfg=norm_cfg,
        device=device,
    )

    return conv


def Linear(
    in_features,
    out_features,
    bias: bool = False,
    device: Device = torch.device("cuda"),
    linear_cfg: dict = dict(type="QLinear"),
    norm_cfg=None,
    act_cfg=None,
    dropout: float = 0.0,
):
    linear = LinearBlock(
        in_features,
        out_features,
        bias=bias,
        linear_cfg=linear_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        dropout=dropout,
        device=device,
    )

    return linear


def MatMul(
    device: Device = torch.device("cuda"),
    matmul_cfg: dict = dict(type="QMatMul"),
):
    matmul = MatMulBlock(
        matmul_cfg=matmul_cfg,
        device=device,
    )

    return matmul
