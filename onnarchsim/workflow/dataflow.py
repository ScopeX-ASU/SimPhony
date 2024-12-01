import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.modules.utils import _pair, _single, _triple
from torchonn.layers import *

from models.layers.quantized_matmul import *

from .utils import track_layer_sizes

PLACEHOLDER_VALUE = 15
__all__ = [
    "im2col_2d_shape",
    "generate_layer_sizes",
    "loading_factor_calculator",
    "calculate_iterations",
    "cycles",
    "extract_layer_info",
]


def extract_layer_info(model):
    layer_info = {}

    for name, layer in model.named_modules():
        layer_type = layer.__class__.__name__
        # We expect your customized Conv2d layer has kernel_size or in_channels attribute to be properly recognized, otherwise
        # you will need to custmoized this function for your own layer
        if hasattr(layer, "kernel_size") or hasattr(layer, "in_channels"):
            if isinstance(layer, int):  # scalar kernel size assumes to be Conv1d
                dimension = 1
            else:
                dimension = len(layer.kernel_size)
            _ntuple = [_single, _pair, _triple][dimension - 1]
            stride = _ntuple(layer.stride if hasattr(layer, "stride") else 1)
            padding = _ntuple(layer.padding if hasattr(layer, "padding") else 0)
            layer_info[name] = (
                "Conv",
                layer,
                (
                    layer.out_channels,
                    layer.in_channels,
                    *layer.kernel_size,
                    stride,
                    padding,
                ),
                layer_type,
            )

        # We expect your customized Linear layer has in_features or out_features attribute to be properly recognized, otherwise
        # you will need to custmoized this function for your own layer
        elif hasattr(layer, "in_features") or hasattr(layer, "out_features"):
            stride = 1
            padding = 0
            layer_info[name] = (
                "Linear",
                layer,
                (layer.out_features, layer.in_features, stride, padding),
                layer_type,
            )

        # To handle MatMul function properly, we replcaed the original MatMul function with a MatMul module
        # You can always customize this function to handle your own MatMul function
        elif isinstance(layer, QMatMul):  # noqa: F405
            layer_info[name] = (
                "MatMul",
                layer,
                (0, 0, 0, 0),
                layer_type,
            )

    return layer_info


# Important function to determine the layer processing cycles
def im2col_2d_shape(
    W: Optional[Union[Tensor, Tuple[int, int, int, int], Tuple[int, int]]] = None,
    X: Optional[Union[Tensor, Tuple[int, int, int, int]]] = None,
    stride: int = 1,
    padding: int = 0,
    layer: str = "Conv",
    layer_shape: Dict[str, any] = None,
) -> (
    tuple[None, None, None]
    | tuple[tuple[int | Any, int | Any], tuple[int | Any, int | Any], tuple[int]]
):
    if layer == "Linear":
        # Handle linear layer
        if X is not None:
            print(X)
            if isinstance(X, torch.Tensor):
                dims = X.size()[-1]

                flattened_size = torch.prod(torch.tensor(X.size()[:-1])).item()
            else:
                dims = X[-1]
                flattened_size = torch.prod(torch.tensor(X[:-1])).item()

            reshaped_matrix_size_X = (dims, flattened_size)

        else:
            return None, None, None

        if W is not None:
            print(W)
            if isinstance(W, torch.Tensor):
                out_features, in_features = W.size()
            else:
                out_features, in_features = W

            reshaped_matrix_size_W = (out_features, in_features)

        else:
            return None, None, None

        return reshaped_matrix_size_W, reshaped_matrix_size_X, None

    elif layer in {"Conv", "Conv1d", "Conv2d", "Conv3d"}:
        if W is not None:
            if isinstance(W, Tensor):
                W = W.shape
            n_filters, d_filter, *filter_size = W
        else:
            return None, None, None

        if X is not None:
            if isinstance(X, Tensor):
                X = X.shape
            n_x, d_x, *input_size = X

            out_size = [
                (x_size - f_size + 2 * p) // s + 1
                for x_size, f_size, s, p in zip(
                    input_size, filter_size, stride, padding
                )
            ]

            reshaped_matrix_size_X = (
                d_x * np.prod(filter_size),
                n_x * np.prod(out_size),
            )
        else:
            return None, None, None

        reshaped_matrix_size_W = (n_filters, np.prod(W[1:]))

        return reshaped_matrix_size_W, reshaped_matrix_size_X, out_size
    elif layer in {"MatMul"}:
        input_size_1 = layer_shape[0]
        input_size_2 = layer_shape[2]
        last_dim = input_size_1[-1]
        reshaped_matrix_size_X = (
            last_dim,
            math.prod(input_size_1[:-1]),
        )
        reshaped_matrix_size_W = (math.prod(input_size_2) // last_dim, last_dim)
        return reshaped_matrix_size_W, reshaped_matrix_size_X, None


def generate_layer_sizes(
    model: nn.Module,
    layer_info: Dict[str, Tuple[str, nn.Module, Tuple[int, int, int, int, int, int]]],
    input_shape: Tuple[int, int, int, int],
    map_cfgs: Dict[str, Any],
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    ## use register hook technique to extract layer input/output shape
    ## {"conv1": ((..in_shape..), (..out_shape...)), ...}
    layer_shapes, layer_sizes = track_layer_sizes(
        model,
        layer_dict={name: info[1] for name, info in layer_info.items()},
        dummy_input_size=input_shape,
        map_cfgs=map_cfgs,
    )
    print(layer_shapes, layer_sizes)
    for layer_name, (layer_type, _, weight_shape, _) in layer_info.items():
        input_shape = layer_shapes[layer_name][0]

        reshaped_weight_size, reshaped_input_size, _ = im2col_2d_shape(
            W=weight_shape[:-2],
            X=input_shape,
            stride=weight_shape[-2],
            padding=weight_shape[-1],
            layer=layer_type,
            layer_shape=layer_shapes[layer_name],
        )

        if reshaped_input_size is None or reshaped_weight_size is None:
            raise ValueError(
                f"Unable to compute reshaped sizes for layer {layer_name} ({layer_type})"
            )


        layer_shapes[layer_name] = (reshaped_input_size, reshaped_weight_size)

    return layer_shapes, layer_sizes


def loading_factor_calculator(
    weight_rep: str = "full",
    input_rep: str = "full",
    output_rep: str = "full",
) -> Tuple[int, int]:
    if weight_rep not in ["full", "complex", "positive"]:
        assert ValueError(f"Weight representation '{weight_rep}' not supported")

    if input_rep not in ["full", "complex", "positive"]:
        assert ValueError(f"Input representation '{input_rep}' not supported")

    if output_rep not in ["full", "complex", "positive"]:
        assert ValueError(f"Output representation '{output_rep}' not supported")

    # Determine loading factors based on representations
    if weight_rep == "positive" and input_rep == "positive" and output_rep == "full":
        forward_factor_x = 2
        forward_factor_w = 2
    elif (
        weight_rep == "positive"
        and input_rep == "positive"
        and output_rep == "positive"
    ):
        forward_factor_x = 1
        forward_factor_w = 1
    elif weight_rep == "positive" and input_rep == "full" and output_rep == "full":
        forward_factor_x = 1
        forward_factor_w = 2
    elif weight_rep == "full" and input_rep == "positive" and output_rep == "full":
        forward_factor_x = 2
        forward_factor_w = 1
    elif weight_rep == "full" and input_rep == "full" and output_rep == "full":
        forward_factor_x = 1
        forward_factor_w = 1
    else:
        raise NotImplementedError

    return forward_factor_x, forward_factor_w


def calculate_iterations(
    miniblock: List[int],
    dim_map: Dict[str, str] = {"M": None, "N": "height", "D": "width"},
    multi_wavelength: int = 1,
    D: int = 5,
    N: int = 5,
    M: int = 5,
) -> Tuple[int, int, int, Dict[str, int], Dict[str, int]]:
    arch_dims = {
        "width": miniblock[3] * miniblock[1],
        "height": miniblock[2] * miniblock[0],
        "wavelength": multi_wavelength,
    }

    core_dims = {
        "width": miniblock[3],
        "height": miniblock[2],
        "wavelength": multi_wavelength,
    }

    iter_D = math.ceil(D / arch_dims[dim_map["D"]]) if dim_map["D"] is not None else D
    iter_N = math.ceil(N / arch_dims[dim_map["N"]]) if dim_map["N"] is not None else N
    iter_M = math.ceil(M / arch_dims[dim_map["M"]]) if dim_map["M"] is not None else M

    return iter_N, iter_M, iter_D, core_dims, arch_dims


# def calculate_required_memory_bandwidth(
#     arch_dims: Dict[str, int],
#     dim_map: Dict[str, str],
#     in_bits: int,
#     w_bits: int,
#     out_bits: int,
#     work_freq: int,
# ) -> float:
#     partition_N = arch_dims[dim_map["N"]] if dim_map["N"] is not None else 1
#     partition_M = arch_dims[dim_map["M"]] if dim_map["M"] is not None else 1
#     partition_D = arch_dims[dim_map["D"]] if dim_map["D"] is not None else 1

#     # One cycle input, weight, and output data under current working frequency
#     # work freq in GHz scale, 1/work_freq in ns
#     # 1e9 to convert to Hz
#     # Required bandwidth in bits/s
#     required_bandwidth = (
#         partition_N * partition_M * out_bits
#         + partition_N * partition_D * w_bits
#         + partition_M * partition_D * in_bits
#     ) / (1 / work_freq / 1e9)

#     return required_bandwidth


def cycles(
    miniblock: List[int],
    matrices: List[Tuple[int, int]],
    layer_sizes: Dict[str, Any],
    dataflow: str = "weight_stationary",
    op_type: str = "MVM",
    dim_map: Dict[str, str] = {"M": None, "N": "height", "D": "width"},
    multi_wavelength: int = 1,
    work_freq: int = 5,  ## In GHz scale
    forward_type: str = "direct",
    weight_representation: str = "full",
    input_representation: str = "full",
    output_representation: str = "full",
) -> Tuple[int, int, int, int, int, int]:
    # matrices[0] is input shape and matrices[1] is weight shape
    N, D1 = matrices[1]  # Weight Shape
    D2, M = matrices[0]  # Input Shape

    print(matrices[1], matrices[0])

    assert D1 == D2, "Got incorrect matrix dimension."

    if dataflow not in ["weight_stationary", "input_stationary", "output_stationary"]:
        assert ValueError(f"Dataflow '{dataflow}' not supported")

    if op_type not in ["VV", "MM", "MVM"]:
        assert ValueError(f"Multiplication type '{op_type}' not supported")

    valid_values = {"width", "height", "wavelength"}
    valid_keys = {"M", "N", "D"}

    for key, value in dim_map.items():
        if key not in valid_keys:
            raise ValueError(
                f"Invalid key '{key}' in dimension mapping. Valid keys are: {valid_keys}"
            )
        if value is not None:
            if value not in valid_values:
                raise ValueError(
                    f"Invalid value '{value}' for key '{key}' in dimension mapping. Valid values are: {valid_values}"
                )

    forward_factor_x, forward_factor_w = loading_factor_calculator(
        weight_rep=weight_representation,
        input_rep=input_representation,
        output_rep=output_representation,
    )

    iter_N, iter_M, iter_D, core_dims, arch_dims = calculate_iterations(
        miniblock=miniblock,
        dim_map=dim_map,
        multi_wavelength=multi_wavelength,
        D=D1,
        N=N,
        M=M,
    )

    return (
        iter_N,
        iter_D,
        iter_M,
        N,
        D1,
        M,
        forward_factor_w,
        forward_factor_x,
        arch_dims,
    )
