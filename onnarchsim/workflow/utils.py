"""
Date: 2024-01-04 19:46:37
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-01-04 19:59:07
FilePath: /ONNArchSim/src/database.py
"""

import math
import re
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from mmengine.registry import Registry
from pyutils.general import logger as lg

# from copy import deepcopy
# from onnarchsim.workflow.energy import reuse_determinator

__all__ = [
    "find_sub_arch",
    "check_layer_mapping",
    "load_device_and_count",
    "load_required_devices",
    "track_layer_sizes",
    "parse_device_name",
    "evaluate_factors",
]

# Create a registry for functions
COUNT_FUNCTIONS = Registry("count_functions")


def parse_device_name(device: str):
    """
    Parses the device name into its components, considering that the type may contain underscores.

    Args:
        device: The device name string in the format 'part_type_index_instance-name'.

    Returns:
        A tuple containing (part, type, index, instance).
    """
    # Split at the first hyphen to separate the identifier and name
    part_type_index_instance, name = device.split("-", 1)

    # Split the part_type_index_instance string into part, type, index, and instance
    # We assume the format is 'part_<type>_index_instance', where <type> can contain underscores

    # Split from the right to get the last two expected parts: index and instance
    split_right = part_type_index_instance.rsplit("_", 2)

    # Ensure that we got exactly three parts
    if len(split_right) != 3:
        raise ValueError(f"Unexpected device format: {device}")

    # Extract part, index, and instance from the split
    index, instance = split_right[1], split_right[2]

    # Everything before the last two underscores is considered the type
    part_and_type = split_right[0]

    # Extract part and type by splitting from the left, assuming 'part' always comes first
    part, type = part_and_type.split("_", 1)
    # print(part, type, index, instance, name)
    return part, type, index, instance, name


def evaluate_factors(
    expression: str,
    R: int,
    C: int,
    core_height: int,
    core_width: int,
    num_wavelength: int = 1,
) -> float | int:
    """
    Evaluates a user-provided expression for the sharing factor, replacing R, H, C, W
    with their respective numerical values.

    Parameters:
    - expression (str): The input string to evaluate.
    - R (int): Numerical value for tiles.
    - C (int): Numerical value for cores per tile.
    - H (int): Numerical value for core height.
    - W (int): Numerical value for core width.
    - num_wavelength (int): Numerical value for number of wavelengths.

    Returns:
    - float: Evaluated result of the expression.
    """
    if type(expression) is int:
        return expression

    # Replace R, H, C, W with their corresponding numerical values
    expression = re.sub(r"\bR\b", str(R), expression)
    expression = re.sub(r"\bH\b", str(core_height), expression)
    expression = re.sub(r"\bC\b", str(C), expression)
    expression = re.sub(r"\bW\b", str(core_width), expression)
    expression = re.sub(r"\bN\b", str(num_wavelength), expression)

    # Allowed functions for evaluation
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "log2": math.log2,
        "log10": math.log10,
        "pow": math.pow,
        "abs": abs,
        "ceil": math.ceil,
        "floor": math.floor,
        "trunc": math.trunc,
        "factorial": math.factorial,
        "max": max,
        "min": min,
        # Add more math functions if needed
    }

    try:
        # Evaluate the expression in a restricted environment
        result = eval(expression, {"__builtins__": None}, allowed_names)
    except (SyntaxError, ZeroDivisionError, NameError, TypeError) as e:
        raise ValueError(f"Invalid expression provided: {e}")

    return int(result)


def find_sub_arch(hetero_arch, name) -> Dict[str, Any]:
    """Find the sub-architecture with the given name."""
    sub_archs = hetero_arch.get("hetero_arch", {}).get("sub_archs", {})
    print(sub_archs)
    for sub_arch in sub_archs.values():
        if sub_arch["name"] == name:
            return sub_arch
    # print(architecture['architecture'])
    # exit(0)
    return None


def check_layer_mapping(
    hetero_arch_config: Dict[str, Any],  # Entire hetero arch config
    model_to_arch_config: Dict[str, Any],  # Mapping of model layer to architecture
    model,  # Entire model
):
    dim_map_cfgs = dict()
    layer_sub_arch_mapping = {}
    miniblock = {}
    sub_arch_to_layer_mapping = {}  # Dictionary to map architecture to list of layers
    sub_arch_config_dict = {}  # Dictionary to store entire architecture
    for layer_name, layer in model.named_modules():

        layer_type = layer.__class__.__name__

        if layer_type in model_to_arch_config:
            arch_name = model_to_arch_config[layer_type]["sub_arch"]
            sub_arch = find_sub_arch(hetero_arch_config, arch_name)
            if sub_arch is None:
                raise ValueError(
                    f"Sub-architecture {arch_name} not found in the architecture config."
                )

            # Save the architecture configuration for the layer
            # Each layer is been mapped to a sub-architecture with detailed entire sub arch configs
            layer_sub_arch_mapping[layer_name] = sub_arch

            # Save the architecture configuration for the layer
            # Add the layer to the arch_to_layer_mapping
            # Here we only have sub arch names for layers uses the same sub arch
            if arch_name not in sub_arch_to_layer_mapping:
                sub_arch_to_layer_mapping[arch_name] = []
            sub_arch_to_layer_mapping[arch_name].append(layer_name)

            # Save the entire sub-architecture config in arch_config_dict
            if arch_name not in sub_arch_config_dict:
                sub_arch_config_dict[arch_name] = sub_arch

            core_precision = sub_arch["core"]["precision"]

            partition_size = [
                sub_arch["tiles"],
                sub_arch["cores_per_tile"],
                sub_arch["core"]["height"],
                sub_arch["core"]["width"],
            ]

            check_attributes = ["miniblock", "w_bit", "in_bit", "out_bit"]

            for attr in check_attributes:
                model_attr = getattr(layer, attr, None)
                if model_attr is None:
                    lg.warning(
                        f"Attribute {attr} not found in layer {layer_name}({layer_type})"
                    )
                    continue

                arch_attr = None
                if attr in core_precision:
                    arch_attr = core_precision[attr]
                elif attr == "miniblock":
                    arch_attr = partition_size

                if model_attr != arch_attr:
                    lg.warning(
                        f"Mismatch in {attr} for layer {layer_name} ({layer_type}): model has {model_attr}, arch has {arch_attr}"
                    )
            miniblock[layer_name] = partition_size
            dim_map_cfgs[layer_name] = model_to_arch_config[layer_type]["dim_map"]

        else:
            lg.info(
                f"Couldn't find the mapping for layer {layer_name} ({layer_type}) in the mapping config."
            )

    lg.info("Layer mapping check completed.")
    return (
        miniblock,
        dim_map_cfgs,
        layer_sub_arch_mapping,
        sub_arch_to_layer_mapping,
        sub_arch_config_dict,
    )


#########################################
## Customized device counter functions ##
#########################################
@COUNT_FUNCTIONS.register_module()
def default_device_counter(arch_config, scaling_rules):
    core_width = arch_config["core"]["width"]
    core_height = arch_config["core"]["height"]
    R = arch_config["tiles"]
    C = arch_config["cores_per_tile"]
    num_wavelength = arch_config["core"].get("num_wavelength", 1)
    if scaling_rules is None:
        return core_width * core_height * R * C
    else:
        return evaluate_factors(
            scaling_rules, R, C, core_height, core_width, num_wavelength
        )
    # return core_width * core_height * R * C


# Most cases can be realized by carefully define the scaling rule
# If not, we can define a new function to handle the case
@COUNT_FUNCTIONS.register_module()
def clements_mzi_uv_counter(arch_config, **kwargs):
    core_width = arch_config.core.width
    core_height = arch_config.core.height
    R = arch_config.tiles
    C = arch_config.cores_per_tile

    m = core_width if kwargs.get("mapping") == "core_width" else core_height
    return (m * (m - 1) / 2) * R * C


@COUNT_FUNCTIONS.register_module()
def clements_mzi_mean_counter(arch_config, **kwargs):
    core_width = arch_config.core.width
    core_height = arch_config.core.height
    R = arch_config.tiles
    C = arch_config.cores_per_tile

    m = min(core_width, core_height)
    return m * R * C


#########################################
## Customized device counter functions ##
#########################################


def device_counter_dispatcher(
    arch_config: Dict[str, Any],
    device_counter_config: Dict[str, Any],
    instance: str,
    scaling_rules: str = None,
) -> int:
    if device_counter_config:
        for group_name, group_info in device_counter_config.items():
            instance_groups = group_info.get("instance_groups", [])
            func_name = group_info.get("func_name")
            func_args = group_info.get("func_args", {})

            # If instance is in the current group's instance list, call the corresponding function
            if instance in instance_groups:
                # Dynamically call the function by name
                func = COUNT_FUNCTIONS.get(func_name)
                if func and callable(func):
                    # Call the function with arch_config and any additional arguments
                    result = func(arch_config, **func_args)
                    return result
                else:
                    raise ValueError(
                        f"Function '{func_name}' not found or is not callable."
                    )

    # If no matching group found or device counter config is none, call the default function
    result = default_device_counter(arch_config, scaling_rules)
    return result


def parse_chip_mappings(instance, chip_mappings: Dict[str, Any]) -> str:
    if chip_mappings is None:
        return "off_chip"
    for chip, instances in chip_mappings.items():
        if instance in instances:
            return chip
    return "off_chip"


def load_device_and_count(
    arch_config: Dict[str, Any],
    config: Dict[str, Any],
    device_db: Dict[str, Dict[str, Any]],
    prefix: str,
) -> Tuple[Dict[str, Any], list, list, Dict[str, Any]]:
    components = {}
    devices = config["devices"]
    scaling_rules = config["netlist"].get("scaling_rules", None)
    chip_mappings = config["netlist"].get("chip_mapping", {})

    operand_1_device = []
    operand_1_encoding_devices = config["netlist"].get("operand_1", {})

    operand_2_device = []
    operand_2_encoding_devices = config["netlist"].get("operand_2", {})

    temporal_shared_device = {}
    temporal_factor_config = config["netlist"].get("temporal_accum_factor", {})
    temporal_effected_devices = temporal_factor_config.get("devices", [])
    temporal_factor = temporal_factor_config.get("duration", 1)

    # Count instances and load power for each device type
    for instance, device_choice in config["netlist"]["instances"].items():
        device_type, device_index = device_choice
        chip_type = parse_chip_mappings(instance, chip_mappings)
        if device_type in devices:
            device_name = devices[device_type][int(device_index)]

            device_final_name = (
                f"{prefix}_{device_type}_{device_index}_{instance}-{device_name}"
            )

            device_config = device_db.get(device_type, {}).get(device_name, {})

            cfgs = device_config.get("cfgs", {})
            static_power = cfgs.get("static_power", 0)
            dynamic_power = cfgs.get("power", cfgs.get("dynamic_power", 0))

            if "area" in cfgs:
                area = cfgs["area"]
            elif "width" in cfgs and "length" in cfgs:
                area = cfgs["width"] * cfgs["length"]
            elif "width" in cfgs or "length" in cfgs:
                area = 0
            else:
                raise ValueError(f"Area not found for device {device_name}")

            # Dealing with sharing factors

            scaling_rule_ins = (
                scaling_rules.get(instance, None) if scaling_rules else None
            )
            count = device_counter_dispatcher(
                arch_config,
                config["device_counter_function"],
                instance,
                scaling_rule_ins,
            )

            # Dealing with different types of PTC's number loading

            # Update count
            if device_final_name not in components:
                components[device_final_name] = {
                    "count": count,
                    "static_power": static_power,
                    "dynamic_power": dynamic_power,
                    "area": area,
                    "chip_type": chip_type,
                    "cfgs": cfgs,
                }
            else:
                components[device_final_name]["count"] += count

            # Dealing with operand 1 devices
            # print(operand_1_encoding_devices)
            if (
                operand_1_encoding_devices is not None
                and instance in operand_1_encoding_devices
            ):
                operand_1_device.append(device_final_name)
            # Dealing with operand 2 devices
            if (
                operand_2_encoding_devices is not None
                and instance in operand_2_encoding_devices
            ):
                operand_2_device.append(device_final_name)
            # Dealing with temporal shared devices
            if (
                temporal_effected_devices is not None
                and instance in temporal_effected_devices
            ):
                temporal_shared_device.update({device_final_name: temporal_factor})
        # TODO: Support multi nodes per core
        elif prefix == "core" and device_type == "node":
            components[device_type] = {}
            components[device_type]["instance"] = instance
            components[device_type]["chip_type"] = parse_chip_mappings(
                instance, chip_mappings
            )
        else:
            raise ValueError(f"Device type {device_type} not match in config devices")

    return components, operand_1_device, operand_2_device, temporal_shared_device


def load_required_devices(
    config: Dict[str, Any],
    device_db: object,
    # dim_map: Dict[str, Optional[str]],
) -> Tuple[Dict[str, Any], list, list, Dict[str, Any]]:
    # Load core devices
    core_config = config.get("core", {})
    print(core_config)
    components, operand_1_devices, operand_2_devices, temporal_shared_devices = (
        load_device_and_count(config, core_config, device_db, "core")
    )
    # core_device_names = list(components.keys())

    # Currently only support one node for each core
    # TODO: Support multiple nodes for each core
    node_config = config.get("core", {}).get("node", {})
    (
        temp_components,
        temp_operand_1_devices,
        temp_operand_2_devices,
        temp_temporal_shared_devices,
    ) = load_device_and_count(config, node_config, device_db, "node")
    # node_device_names = list(node_components.keys())
    components.update(temp_components)
    operand_1_devices.extend(temp_operand_1_devices)
    operand_2_devices.extend(temp_operand_2_devices)
    temporal_shared_devices.update(temp_temporal_shared_devices)

    return components, operand_1_devices, operand_2_devices, temporal_shared_devices


class LayerSizeTracker(object):
    def __init__(self):
        self.layer_shape = {}
        self.layer_sizes = (
            {}
        )  # Stores actual sizes (in bytes) of weights, inputs, and outputs

    def get_size(self, module, input, output):
        if "matmul" in module.__class__.__name__.lower():
            self.layer_shape[module._temp_track_name_] = (
                tuple(input[0].shape),
                tuple(output.shape),
                tuple(input[1].shape),
            )
        else:
            self.layer_shape[module._temp_track_name_] = (
                tuple(input[0].shape),
                tuple(output.shape),
            )

    def calculate_size(self, module, mode, in_bits, w_bits, out_bits):
        layer_name = module._temp_track_name_

        input_shape = self.layer_shape[layer_name][0]

        num_input_elements = torch.prod(torch.tensor(input_shape)).item()

        input_size = (num_input_elements * in_bits) / 8  # Convert bits to bytes

        # Calculate output size
        output_shape = self.layer_shape[layer_name][1]
        num_output_elements = torch.prod(torch.tensor(output_shape)).item()
        output_size = (num_output_elements * out_bits) / 8

        if len(self.layer_shape[layer_name]) == 3:
            weight_shape = self.layer_shape[layer_name][2]
            num_weight_elements = torch.prod(torch.tensor(weight_shape)).item()
            weight_size = (num_weight_elements * w_bits) / 8  # Convert bits to bytes
        else:
            # Calculate weight size if weights are available
            weight_size = 0
            if hasattr(module, "weights") and module.weights is not None:
                # for mode, weight_info in module.weights.items():
                for weight in module.weights[mode]:
                    weight_size += weight.numel() * w_bits / 8
            elif hasattr(module, "weight") and module.weight is not None:
                weight_size = module.weight.numel() * w_bits / 8

        self.layer_sizes[layer_name] = {
            "input_size": [input_size, in_bits],
            "output_size": [output_size, out_bits],
            "weight_size": [weight_size, w_bits],
            "total_size": input_size + output_size + weight_size,
        }


# Track layer input output shape
# Extract self.weights with the desired mode
# Also check corresponding quantizer for resolution
def track_layer_sizes(
    model: nn.Module,
    layer_dict: dict = {},
    dummy_input_size: Tuple = (1, 3, 32, 32),
    map_cfgs: Dict[str, Any] = None,
):
    # Function to find and match the layer type with config
    def match_layer_with_config(layer, config):
        # Get the actual type of the layer
        layer_type = layer.__class__.__name__

        # Search through the config to find the matching type
        for key, cfg in config.items():
            if cfg.get("type") == layer_type:
                return cfg
        return None

    tracker = LayerSizeTracker()
    hooks = []
    for name, layer in layer_dict.items():
        layer._temp_track_name_ = name
        hook = layer.register_forward_hook(tracker.get_size)
        hooks.append(hook)

    device = next(model.parameters()).device
    dummy_input = torch.randn(dummy_input_size).to(device)
    is_train = model.training
    model.train()  # not eval mode to initialize quantizers, otherwise it will raise error
    model = model.to(device)
    with torch.no_grad():
        model(dummy_input)
    model.train(is_train)

    for hook in hooks:
        hook.remove()

    # Calculate actual sizes based on bit precision
    for name, layer in layer_dict.items():
        layer_cfgs = match_layer_with_config(layer, map_cfgs)
        layer_mode = layer_cfgs.get("mode", "phase") if layer_cfgs else "phase"
        layer_in_bits = layer_cfgs.get("in_bit", 8) if layer_cfgs else 8
        layer_w_bits = layer_cfgs.get("w_bit", 8) if layer_cfgs else 8
        layer_out_bits = layer_cfgs.get("out_bit", 8) if layer_cfgs else 8
        tracker.calculate_size(
            layer, layer_mode, layer_in_bits, layer_w_bits, layer_out_bits
        )

    for layer in layer_dict.values():
        delattr(layer, "_temp_track_name_")
    torch.cuda.empty_cache()

    return tracker.layer_shape, tracker.layer_sizes
