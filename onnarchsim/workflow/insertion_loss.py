from typing import Any, Dict, List, Tuple

import networkx as nx

from onnarchsim.workflow.utils import evaluate_factors

__all__ = [
    "construct_node_graph",
    "construct_core_graph",
    "architecture_insertion_loss",
]


def get_instance_insertion_loss(
    device_choice: Dict[str, List[str]],
    devices: Dict[str, List[str]],
    device_db: object,
) -> str:
    device_type, index = device_choice
    if device_type in devices:
        device_name = devices[device_type][int(index)]
    else:
        raise ValueError(f"Device type {device_type} not match in config devices")
    if device_name in device_db[device_type]:
        il = (
            device_db[device_type][device_name]["cfgs"]["insertion_loss"]
            if hasattr(device_db[device_type][device_name]["cfgs"], "insertion_loss")
            else 0
        )

        return il
    else:
        raise ValueError(f"Device name {device_name} not found in device database")


def get_replication_count(
    instance: str,
    device_replication: Dict[str, str],
    sub_arch_dims: Tuple[int, int, int, int, int],
) -> int:
    R, C, H, W, N = sub_arch_dims
    if device_replication is not None:
        if instance in device_replication:
            expr = device_replication[instance]
            return evaluate_factors(expr, R, C, H, W, N)
    return 1


def parse_node(node_str, replication_counts):
    if "(" in node_str:
        instance_id, idx = node_str.split("(")
        idx = idx.strip(")")
        if idx == "s":
            index = 0  # Start index
        elif idx == "e":
            index = replication_counts[instance_id] - 1  # End index
        else:
            index = int(idx)
        return instance_id, f"{instance_id}_{index}"
    else:
        instance_id = node_str
        return instance_id, instance_id


def construct_node_graph(
    node_config: Dict[str, List[str]],
    device_db: Dict[str, Dict[str, Any]],
    sub_arch_dims: Tuple[int, int, int, int, int],
) -> Tuple[nx.DiGraph, Dict[str, float]]:
    # Initialize weights and nets
    instances = node_config["netlist"]["instances"]
    devices = node_config["devices"]
    nets = node_config["netlist"]["nets"]
    node_device_replication = node_config.get("netlist", {}).get("replication", {})

    # print(node_device_replication)
    # exit(0)
    instance_weights = {}
    replication_counts = {}
    for instance, device_choice in instances.items():
        replication_counts[instance] = get_replication_count(
            instance, node_device_replication, sub_arch_dims
        )
        instance_weights[instance] = (
            get_instance_insertion_loss(device_choice, devices, device_db)
            if device_db
            else 0
        )

    G = nx.DiGraph()

    # Add nodes to the graph
    for instance_id, replication_count in replication_counts.items():
        if replication_count > 1:
            for idx in range(replication_count):
                node_name = f"{instance_id}_{idx}"
                G.add_node(node_name)
        else:
            G.add_node(instance_id)

    # Add edges to replicated nodes
    for instance_id, replication_count in replication_counts.items():
        if replication_count > 1:
            for idx in range(replication_count - 1):
                src = f"{instance_id}_{idx}"
                dst = f"{instance_id}_{idx+1}"
                G.add_edge(src, dst, weight=instance_weights[instance_id])

    # Add edges to the graph
    for net, (src, dst) in nets.items():
        src_parsed_id, src_node_name = parse_node(src, replication_counts)
        dst_parsed_id, dst_node_name = parse_node(dst, replication_counts)
        if src_parsed_id in instance_weights and dst_parsed_id in instance_weights:
            G.add_edge(
                src_node_name, dst_node_name, weight=instance_weights[dst_parsed_id]
            )
        else:
            raise ValueError(
                f"Source {src} or destination {dst} not match with defined instances"
            )


    return G, instance_weights, instances


def construct_core_graph(
    core_config: Dict[str, List[str]],
    device_db: Dict[str, Dict[str, Any]],
    node_insertion_loss: float,
    sub_arch_dims: Tuple[int, int, int, int, int],
) -> Tuple[nx.DiGraph, Dict[str, float]]:
    # Initialize weights and nets
    instances = core_config["netlist"]["instances"]
    devices = core_config["devices"]
    nets = core_config["netlist"]["nets"]
    core_device_replication = core_config.get("netlist", {}).get("replication", {})
    instance_weights = {}
    replication_counts = {}

    for instance, device_choice in instances.items():
        if device_choice[0] == "node":
            instance_weights[instance] = node_insertion_loss
        else:
            instance_weights[instance] = (
                get_instance_insertion_loss(device_choice, devices, device_db)
                if device_db
                else 0
            )
        replication_counts[instance] = get_replication_count(
            instance, core_device_replication, sub_arch_dims
        )

    G = nx.DiGraph()

    # Add nodes to the graph
    for instance_id, replication_count in replication_counts.items():
        if replication_count > 1:
            for idx in range(replication_count):
                node_name = f"{instance_id}_{idx}"
                G.add_node(node_name)
        else:
            G.add_node(instance_id)

    # Add edges to replicated nodes
    for instance_id, replication_count in replication_counts.items():
        if replication_count > 1:
            for idx in range(replication_count - 1):
                src = f"{instance_id}_{idx}"
                dst = f"{instance_id}_{idx+1}"
                G.add_edge(src, dst, weight=instance_weights[instance_id])

    # Add edges to the graph
    for net, (src, dst) in nets.items():
        src_parsed_id, src_node_name = parse_node(src, replication_counts)
        dst_parsed_id, dst_node_name = parse_node(dst, replication_counts)
        if src_parsed_id in instance_weights and dst_parsed_id in instance_weights:
            G.add_edge(
                src_node_name, dst_node_name, weight=instance_weights[dst_parsed_id]
            )
        else:
            raise ValueError(
                f"Source {src} or destination {dst} not match with defined instances"
            )

    return G, instance_weights, instances


def find_longest_path(
    G: nx.DiGraph,
    instance_weights: Dict[str, float],
    instances: Dict[str, List[str]],
    target_device_type: str = "photodetector",
) -> Tuple[List[str], float]:
    topo_order = list(nx.topological_sort(G))
    dist = {node: float("-inf") for node in G}
    path = {node: [] for node in G}

    # start_node = topo_order[0]
    # dist[start_node] = 0
    start_nodes = [node for node in G if G.in_degree(node) == 0]
    for start_node in start_nodes:
        node_id = start_node.split("_")[0] if "_" in start_node else start_node
        dist[start_node] = instance_weights[node_id]

    for u in topo_order:
        if dist[u] != float("-inf"):
            for v in G.neighbors(u):
                weight = G[u][v]["weight"]
                if dist[v] < dist[u] + weight:
                    dist[v] = dist[u] + weight
                    path[v] = path[u] + [u]

    # Filter to consider only paths ending at a target device
    max_insertion_loss = float("-inf")
    longest_path = []

    # Find all nodes that are target device
    target_device = [
        node for node, device in instances.items() if device[0] == target_device_type
    ]

    max_insertion_loss = float("-inf")
    longest_path = []

    for target in target_device:
        target_nodes = [
            node for node in dist if node == target or node.startswith(f"{target}_")
        ]
        for target_node in target_nodes:
            if dist[target_node] > max_insertion_loss:
                max_insertion_loss = dist[target_node]
                longest_path = path[target_node] + [target_node]
            else:
                continue
    return longest_path, max_insertion_loss


def architecture_insertion_loss(
    sub_arch: Dict[str, List[str]],
    device_db: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], float]:
    R = sub_arch.get("tiles", {})
    C = sub_arch.get("cores_per_tile", {})
    core_config = sub_arch.get("core", {})
    node_config = sub_arch.get("core", {}).get("node", {})
    H = core_config.get("height", {})
    W = core_config.get("width", {})
    N = core_config.get("num_wavelength", {})
    sub_arch_dims = (R, C, H, W, N)
    core_target_device = core_config.get("netlist", {}).get(
        "insertion_loss_final_device", None
    )
    node_target_device = node_config.get("netlist", {}).get(
        "insertion_loss_final_device", None
    )

    node_graph, node_instance_weights, node_instances = construct_node_graph(
        node_config, device_db, sub_arch_dims
    )
    node_longest_path, node_insertion_loss = find_longest_path(
        node_graph, node_instance_weights, node_instances, node_target_device
    )

    core_graph, core_instance_weights, core_instances = construct_core_graph(
        core_config, device_db, node_insertion_loss, sub_arch_dims
    )

    core_longest_path, core_insertion_loss = find_longest_path(
        core_graph, core_instance_weights, core_instances, core_target_device
    )

    return (
        node_longest_path,
        node_insertion_loss,
        core_longest_path,
        core_insertion_loss,
    )
