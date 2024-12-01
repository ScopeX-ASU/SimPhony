from typing import Union

import torch
from torch import nn
from torch.types import Device, _size

from models.layers.utils import *  # noqa: F403

from .quantized_base import ConvBlock, LinearBlock, MatMulBlock, QBaseModel
from .quantized_mobileViT import QTransformerEncoderLayer

__all__ = [
    "QBertViTBase",
]

class ConvolutionalEmbedding(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        conv_cfg: dict = dict(type="QConv2d"),
        act_cfg=None,
        norm_cfg=None,
        device: Device = torch.device("cuda"),
    ):
        super(ConvolutionalEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Convolutional layer for patch embedding
        self.proj = ConvBlock(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            device=device,
        )

        # Store embedding dimension for mapping
        self.embedding_dim = embed_dim

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class QBertViTBase(QBaseModel):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        conv_cfg=dict(type="QConv2d"),
        linear_cfg=dict(type="QLinear"),
        matmul_cfg=dict(type="QMatMul"),
    ):
        super(QBertViTBase, self).__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = int(embed_dim * mlp_ratio)
        self.device = device

        # Patch Embedding
        self.patch_embed = ConvolutionalEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=None,
            device=device,
        )

        num_patches = self.patch_embed.num_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer Encoder layers
        self.layers = nn.ModuleList(
            [
                QTransformerEncoderLayer(
                    dim=embed_dim,
                    nhead=num_heads,
                    dim_heads=embed_dim // num_heads,
                    mlp_dim=self.mlp_dim,
                    linear_cfg=linear_cfg,
                    matmul_cfg=matmul_cfg,
                    device=device,
                )
                for _ in range(depth)
            ]
        )

        # Final LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [batch_size, in_chans, img_size, img_size]
        batch_size = x.size(0)
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # [batch_size, 1, embed_dim]
        x = torch.cat(
            (cls_tokens, x), dim=1
        )  # [batch_size, num_patches + 1, embed_dim]
        x = x + self.pos_embed  # [batch_size, num_patches + 1, embed_dim]

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x[:, 0]  # Return the class token embedding


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
