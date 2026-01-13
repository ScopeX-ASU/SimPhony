import math
import warnings
from collections import deque
from typing import Any, Dict, List, Tuple

import networkx as nx
from mmengine.registry import Registry

from onnarchsim.workflow.insertion_loss import construct_node_graph
from onnarchsim.workflow.utils import (
    evaluate_factors,
    load_required_devices,
    parse_device_name,
)

__all__ = [
    "area_calculator",
    "chip_area_calculator",
]


BOUNDING_BOX_ESTIMATORS = Registry("bounding_box_estimators")


# Just like in energy device initialization, you can initialize the area of the devices here if the device needs specialized
# method to do so.
# For example, MMI area needs to be scaled based on the port number of the MMI
def cal_MMI_area(
    config: Dict[str, List[str]],
    ports: List[int],
    R: int,
    C: int,
    core_width: int,
    core_height: int,
    num_wavelength: int = 1,
) -> float:
    port = evaluate_factors(ports, R, C, core_height, core_width, num_wavelength)
    port_scaling = port / config["cfgs"]["ports_num"]

    # Currently not supporting input scaling for MMI
    area = config["area"] * port_scaling
    return area


def parsing_config_from_instance(
    provided_instance: str,
    node_config: Dict[str, List[str]],
) -> Dict[str, Any]:
    for device, device_cfg in node_config.items():
        _, _, _, instance, _ = parse_device_name(device)
        if provided_instance == instance:
            # print(device_cfg)
            return device_cfg
    raise ValueError(f"Instance {provided_instance} not found in the node config")


def topological_sort_by_levels_sorted(
    graph: nx.DiGraph, devices: Dict[str, Dict[str, Any]]
) -> list:
    """
    Performs a topological sort on the given DAG, sorting devices within each level by width.

    Args:
        graph: A directed acyclic graph (DAG) represented as a NetworkX DiGraph.
        devices: A dictionary containing device configurations.

    Returns:
        A list of lists where each sublist contains nodes at the same level, sorted by device width.
    """
    # Initialize in-degree for each node
    in_degree = {node: 0 for node in graph.nodes()}
    for u, v in graph.edges():
        in_degree[v] += 1

    # Initialize queue with nodes having zero in-degree
    zero_in_degree = deque([node for node in graph.nodes() if in_degree[node] == 0])
    levels = []  # To store nodes grouped by levels

    while zero_in_degree:
        current_level = []  # Nodes for the current level
        for _ in range(len(zero_in_degree)):
            node = zero_in_degree.popleft()
            current_level.append(node)

            # Decrease in-degree of successors
            for successor in graph.successors(node):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    zero_in_degree.append(successor)

        # Sort the current level by device width in ascending order using parsing_config_from_instance
        sorted_current_level = sorted(
            current_level,
            key=lambda x: parsing_config_from_instance(x, devices)["length"],
        )
        levels.append(sorted_current_level)

    # Flatten the levels into a single sorted 1D list
    sorted_nodes = [node for level in levels for node in level]
    return sorted_nodes


#######################################
## Customized Bounding Box Estimator ##
#######################################


def default_bounding_box_estimator(
    arch_config: Dict[str, List[str]],
    devices: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    node_config = arch_config["core"]["node"]
    core_width = arch_config["core"]["width"]
    core_height = arch_config["core"]["height"]
    R = arch_config["tiles"]
    C = arch_config["cores_per_tile"]

    sub_arch_dim = (R, C, core_height, core_width, 1)
    node_spacing_flag = (
        node_config["node_spacing_flag"]
        if "node_spacing_flag" in node_config
        else False
    )
    device_spacing_flag = (
        node_config["device_spacing_flag"]
        if "device_spacing_flag" in node_config
        else False
    )
    device_spacing = (
        node_config["device_spacing"]
        if "device_spacing" in node_config
        else [5, 5, 5, 5]
    )
    node_spacing = (
        node_config["node_spacing"] if "node_spacing" in node_config else [5, 5, 5, 5]
    )

    node_devices = {
        device: device_cfg for device, device_cfg in devices.items() if "node" in device
    }

    # Max length of the node
    max_length = max(
        [
            device_cfg["length"]
            for device, device_cfg in node_devices.items()
            if "node" in device
        ]
    )

    max_length = (
        max_length + device_spacing[2] + device_spacing[3]
        if device_spacing_flag
        else max_length
    )

    print(max_length)

    node_graph, _, _ = construct_node_graph(node_config, None, sub_arch_dim)

    node_sort = topological_sort_by_levels_sorted(node_graph, node_devices)

    # Device placement setup
    levels = []  # This will store the placed devices in each level
    current_level = []
    current_level_length = 0  # Current occupied length in this level

    # Retrieve instances and their connections from the graph
    for instance in node_sort:
        device_cfg = parsing_config_from_instance(instance, node_devices)
        device_length = device_cfg["length"]
        device_width = device_cfg["width"]
        print(device_length, device_width)
        # Calculate the new level length with device and optional spacing
        added_length = device_length + (
            device_spacing[2] + device_spacing[3] if device_spacing_flag else 0
        )
        temp_level_length = current_level_length + added_length
        if temp_level_length <= max_length:
            current_level.append(device_width)
            current_level_length = temp_level_length
        else:
            levels.append(current_level)
            current_level = [device_width]
            current_level_length = added_length

    # Append the last level if it's not empty
    if current_level:
        levels.append(current_level)

    # Calculate the total bounding box dimensions
    total_height = 0
    total_width = max_length

    # Calculate dimensions considering zigzag placement and spacing
    for level in levels:
        level_height = max(level)
        level_height += (
            device_spacing[0] + device_spacing[1] if device_spacing_flag else 0
        )  # Add top/bottom spacing
        total_height += level_height  # Add the height of the current level

    # Add node spacing if the flag is set
    if node_spacing_flag:
        total_width += (
            node_spacing[2] + node_spacing[3]
        )  # Add left and right node spacing
        total_height += (
            node_spacing[0] + node_spacing[1]
        )  # Add top and bottom node spacing

    area = total_width * total_height
    count = core_width * core_height * R * C

    return area, total_width, total_height, count


@BOUNDING_BOX_ESTIMATORS.register_module()
def default_area_benchmark(
    sub_arch: Dict[str, List[str]],
    devices: Dict[str, Any],
    **kwargs: Dict[str, Any],
):
    core_width = sub_arch["core"]["width"]
    core_height = sub_arch["core"]["height"]
    R = sub_arch["tiles"]
    C = sub_arch["cores_per_tile"]
    node_devices = {
        device: device_cfg for device, device_cfg in devices.items() if "node" in device
    }

    for device, device_cfg in node_devices.items():
        node_area = (
            device_cfg["area"] * device_cfg["count"] if "area" in device_cfg else 0
        )
        device_cfg["total_area"] = node_area

    total_width = total_height = math.sqrt(node_area)
    count = core_width * core_height * R * C

    return node_area, total_width, total_height, count


# Here is the old implementation of the MZI Mesh bounding box estimator
# TODO: A better implementation for MZI Mesh bounding box estimator
# after support multi-node per core

# @BOUNDING_BOX_ESTIMATORS.register_module()
# def clements_mzi_bounding_box_estimator(
#     arch_config: Dict[str, List[str]],
#     devices: Dict[str, Any],
#     **kwargs: Dict[str, Any],
# ) -> Tuple[float, float, float, float]:

#     node_config = arch_config["core"]["node"]
#     core_width = arch_config["core"]["width"]
#     core_height = arch_config["core"]["height"]
#     R = arch_config["tiles"]
#     C = arch_config["cores_per_tile"]
#     node_spacing_flag = node_config["node_spacing_flag"] if "node_spacing_flag" in node_config else False
#     device_spacing_flag = node_config["device_spacing_flag"] if "device_spacing_flag" in node_config else False
#     device_spacing = node_config["device_spacing"] if "device_spacing" in node_config else [5, 5, 5, 5]
#     node_spacing = node_config["node_spacing"] if "node_spacing" in node_config else [5, 5, 5, 5]

#     node_instances = node_config["netlist"]["instances"]
#     node_devices = {device: device_cfg for device, device_cfg in devices.items() if "node" in device}

#     # Extract U, V, and sigma configurations
#     U, U_mapping = kwargs.get("U", ([], "core_width"))
#     V, V_mapping = kwargs.get("V", ([], "core_height"))
#     mean = kwargs.get("mean", [])

#     # Function to determine width and height of devices in U, V, and sigma
#     def get_device_dimensions(device_list: List[str]) -> Tuple[float, float]:
#         # mzi_width = 0
#         # mzi_height = 0
#         coupler_widths = []
#         phase_shifter_widths = []
#         coupler_heights = []
#         phase_shifter_heights = []
#         # for instance in device_list:
#         #     device_type = node_instances[instance][0]  # Get the device type
#         # Ensure device config exists and fetch width and height
#         for instance in device_list:
#             device_type, _ = node_instances[instance]  # Get the device type
#             device_cfg = parsing_config_from_instance(instance, node_devices) # Parse the device configuration

#             # Ensure device config exists
#             if device_cfg:
#                 if 'coupler' in device_type:
#                     coupler_widths.append(device_cfg.get("length", 0))
#                     coupler_heights.append(device_cfg.get("width", 0))
#                 elif 'phase_shifter' in device_type:
#                     phase_shifter_widths.append(device_cfg.get("length", 0))
#                     phase_shifter_heights.append(device_cfg.get("width", 0))

#         # Calculate MZI dimensions if both device types are present
#         if coupler_widths and phase_shifter_widths:
#             mzi_width = sum(coupler_widths) + max(phase_shifter_widths)
#             mzi_height = max(coupler_heights) + max(phase_shifter_heights)
#             # max_width = max(max_width, mzi_width)
#             # max_height = max(max_height, mzi_height)

#         return mzi_width, mzi_height

#         # Calculate dimensions for sigma
#     mean_base_width, mean_base_height = get_device_dimensions(mean)

#     # Calculate dimensions for U (base dimensions depend on "core_width" or "core_height")
#     U_base_width, U_base_height = get_device_dimensions(U)

#     # Calculate dimensions for V
#     V_base_width, V_base_height = get_device_dimensions(V)

#     mean_final_height = (mean_base_height + (device_spacing[0] + device_spacing[1] if device_spacing_flag else 0)) * min(core_width, core_height)

#     V_final_height = (V_base_height + (device_spacing[0] + device_spacing[1] if device_spacing_flag else 0)) * ((core_height if V_mapping == "core_height" else core_width) - 1)
#     U_final_height = (U_base_height + (device_spacing[0] + device_spacing[1] if device_spacing_flag else 0)) * ((core_height if U_mapping == "core_height" else core_width) - 1)

#     mean_final_width = (mean_base_width + (device_spacing[2] + device_spacing[3] if device_spacing_flag else 0))
#     V_final_width = (V_base_width + (device_spacing[2] + device_spacing[3] if device_spacing_flag else 0)) * (core_height if V_mapping == "core_height" else core_width)
#     U_final_width = (U_base_width + (device_spacing[2] + device_spacing[3] if device_spacing_flag else 0)) * (core_height if U_mapping == "core_height" else core_width)

#     node_final_height = max([mean_final_height, V_final_height, U_final_height])
#     node_final_width = mean_final_width + V_final_width + U_final_width

#     if node_spacing_flag:
#         node_final_width += node_spacing[2] + node_spacing[3]
#         node_final_height += node_spacing[0] + node_spacing[1]

#     node_area = node_final_height * node_final_width

#     count = R * C

#     return node_area, node_final_width, node_final_height, count

#######################################
## Customized Bounding Box Estimator ##
#######################################


def bounding_box_estimator_dispatcher(
    arch_config: Dict[str, Any],
    devices: Dict[str, Any],
    bounding_box_estimator_config: Dict[str, Any],
) -> Tuple[float, float, float]:
    # Extract the function name and arguments from the config
    if bounding_box_estimator_config:
        func_name = bounding_box_estimator_config.get("func_name")
        func_args = bounding_box_estimator_config.get("func_args", {})

        # Check if the function name is provided and exists
        if func_name:
            func = BOUNDING_BOX_ESTIMATORS.get(func_name)
            if func and callable(func):
                try:
                    # Call the function with arch_config and any additional arguments
                    area, node_width, node_length, count = func(
                        arch_config, devices, **func_args
                    )
                    return area, node_width, node_length, count
                except Exception as e:
                    raise RuntimeError(f"Error executing function '{func_name}': {e}")
            else:
                raise ValueError(
                    f"Function '{func_name}' not found or is not callable."
                )

    area, node_width, node_length, count = default_bounding_box_estimator(
        arch_config, devices
    )
    return area, node_width, node_length, count


# Able to calculate the area of nodes in a layout aware manner
def cal_node_params(
    sub_arch: Dict[str, List[str]],
    devices: Dict[str, Any],
) -> Dict[str, Any]:
    core_width = sub_arch["core"]["width"]
    core_height = sub_arch["core"]["height"]
    R = sub_arch["tiles"]
    C = sub_arch["cores_per_tile"]
    node_config = sub_arch["core"]["node"]
    bounding_box_estimator_config = sub_arch["core"]["node"]["bounding_box_estimator"]
    bounding_box_area = sub_arch["core"]["node"].get("bounding_box_area", None)

    if bounding_box_area is not None:
        area = bounding_box_area
        node_width = (
            node_config["node_width"]
            if node_config["node_width"] is not None
            else math.sqrt(area)
        )
        node_length = (
            node_config["node_length"]
            if node_config["node_length"] is not None
            else math.sqrt(area)
        )
        assert (
            abs((node_width * node_length - area) / area) < 0.01
        ), "Area and node width and length are not matching"
        count = core_width * core_height * R * C

    else:
        print("Entered here")
        area, node_width, node_length, count = bounding_box_estimator_dispatcher(
            sub_arch, devices, bounding_box_estimator_config
        )

    return area, node_width, node_length, count


def initalize_all_devices_area(
    sub_arch: Dict[str, Any],
    components: Dict[str, Any],
) -> Dict[str, Any]:
    core_config = sub_arch["core"]
    core_width = sub_arch["core"]["width"]
    core_height = sub_arch["core"]["height"]
    R = sub_arch["tiles"]
    C = sub_arch["cores_per_tile"]
    num_wavelength = sub_arch["core"]["num_wavelength"]

    # TODO: Need to handle multi-node per core
    chip_type_node = components.pop("node")["chip_type"]

    for device, device_cfg in components.items():
        part, type, index, instance, name = parse_device_name(device)
        if "mmi" in type:
            ports_num = core_config.get("netlist", {}).get("ports_num", {})
            area = cal_MMI_area(
                device_cfg,
                ports_num[instance],
                R,
                C,
                core_width,
                core_height,
                num_wavelength,
            )
            device_cfg["area"] = area

        for key in list(device_cfg.keys()):
            device_cfg["width"] = (
                device_cfg["cfgs"]["width"]
                if "width" in device_cfg["cfgs"]
                else math.sqrt(device_cfg["cfgs"]["area"])
            )
            device_cfg["length"] = (
                device_cfg["cfgs"]["length"]
                if "length" in device_cfg["cfgs"]
                else math.sqrt(device_cfg["cfgs"]["area"])
            )
            if key not in ["count", "area", "chip_type"]:
                device_cfg.pop(key)

    # TODO: Need to handle multi-node per core
    node_area, node_width, node_length, count = cal_node_params(sub_arch, components)

    components.update(
        {
            "node": {
                "area": node_area,
                "width": node_width,
                "length": node_length,
                "count": count,
                "chip_type": chip_type_node,
            }
        }
    )

    return components


def area_calculator(
    config: Dict[str, List[str]],
    device_db: object,
) -> Dict[str, Any]:
    components, _, _, _ = load_required_devices(config, device_db)
    devices = initalize_all_devices_area(config, components)

    for device, device_cfg in devices.items():
        device_cfg["total_area"] = device_cfg["area"] * device_cfg["count"]

    return devices


def chip_area_calculator(configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    # Initialize dictionaries to classify nodes by chip type
    chip_classification = {}
    for device, area_config in configs.items():
        chip_type = area_config.get("chip_type", "off_chip")
        if device != "node" and "node" in device:
            if chip_type != "off_chip":
                warnings.warn(
                    f"Node device {device} should not be duplicated in the node area configuration level. Be aware of the wrong chip area calculation."
                )
            else:
                continue
        # Initialize the entry for this chip type if not already present
        if chip_type not in chip_classification:
            chip_classification[chip_type] = {
                "devices": [device],
                "total_area": area_config["total_area"],
            }
        else:
            chip_classification[chip_type]["devices"].append(device)
            chip_classification[chip_type]["total_area"] += area_config["total_area"]

    return chip_classification
