from typing import Optional

import torch
from mmengine.registry import MODELS
from torch import Tensor, nn
from torch.types import Device

from .quantized_linear import QLinear
from .quantized_matmul import QMatMul

__all__ = [
    "QAttention",
]


@MODELS.register_module()
class QAttention(nn.Module):
    """
    Quantized Attention.
    """

    __constants__ = [
        "num_heads",
        "dim",
        "head_dim",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    num_heads: int
    dim: int
    head_dim: int

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        w_bit: int = 32,
        in_bit: int = 32,
        out_bit: int = 32,
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        super().__init__(device=device)
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.query = QLinear(
            in_features=dim,
            out_features=dim,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            bias=False,
            device=device,
        )

        self.key = QLinear(
            in_features=dim,
            out_features=dim,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            bias=False,
            device=device,
        )

        self.value = QLinear(
            in_features=dim,
            out_features=dim,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            bias=False,
            device=device,
        )

        self.out = QLinear(
            in_features=dim,
            out_features=dim,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            bias=False,
            device=device,
        )

        self.quantized_matmul = QMatMul(
            w_bit=w_bit,
            in_bit=in_bit,
            out_bit=out_bit,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.size()

        q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = self.quantized_matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / (self.head_dim**0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        attn_output = self.quantized_matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)

        output = self.out(attn_output)
        return output

    def extra_repr(self):
        s = "Quantized_Attention"
        return s


class QTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        w_bit: int = 32,
        in_bit: int = 32,
        out_bit: int = 32,
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        super(QTransformerEncoderLayer, self).__init__()
        self.self_attn = QAttention(
            d_model,
            nhead,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            device=device,
        )
        self.linear1 = QLinear(
            d_model,
            dim_feedforward,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            device=device,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(
            dim_feedforward,
            d_model,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            device=device,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src
