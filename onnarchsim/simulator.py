"""
Date: 2024-08-03 13:38:23
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-08-03 14:12:40
FilePath: /ONNArchSim/onnarchsim/simulator.py
"""

from typing import List

import yaml
from pyutils.general import get_logger
from torch import nn

from onnarchsim.database.device_db import DeviceLib
from onnarchsim.database.hetero_arch_db import HeteroArchitectureLib
from onnarchsim.database.utils import break_path, ensure_file_exists
from onnarchsim.version import __version__
from onnarchsim.workflow.area import area_calculator, chip_area_calculator
from onnarchsim.workflow.dataflow import (
    cycles,
    extract_layer_info,
    generate_layer_sizes,
)
from onnarchsim.workflow.energy import chip_energy_calculator, energy_calculator
from onnarchsim.workflow.insertion_loss import architecture_insertion_loss
from onnarchsim.workflow.memory import (
    calculate_memory_latency_and_energy,
    generate_memory_setting,
)
from onnarchsim.workflow.utils import check_layer_mapping
from utils.config import Config

try:
    from torchonn.models.base_model import ONNBaseModel
except Exception:
    ONNBaseModel = None
    print("Warning: torchonn package not found")

logger = None


class ONNArchSimulator(object):
    def __init__(
        self,
        nn_model: nn.Module = None,
        onn_conversion_cfg: str = "configs/onn_mapping/simple_cnn.yml",
        nn_conversion_cfg: str = "configs/nn_mapping/simple_cnn.yml",
        onn_model: nn.Module = None,
        model2arch_map_cfg: str = "configs/architecture_mapping/simple_cnn.yml",
        devicelib_root: str = "configs/devices",
        device_cfg_files: str | List[str] = ["*/*.yml"],
        arch_cfg_file: str = "configs/design/architectures/TeMPO_hetero.yml",
        arch_version: str = "v1",
        input_shape: tuple = (1, 1, 8, 8),
        log_path="log/test_log.txt",
    ):
        """

        Args:
            nn_model (nn.Module, optional):Neural network model. Defaults to None.
            onn_conversion_cfg (str, optional): NN to Optical neural network model conversion configs. Defaults to "configs/onn_mapping/simple_cnn.yml".
            onn_model (nn.Module, optional): Optical neural network configs. Defaults to None.
            model2arch_map_cfg (str, optional): _description_. Defaults to "configs/architecture_mapping/simple_cnn.yml".
            devicelib_root (str, optional): _description_. Defaults to "configs/devices".
            device_cfg_files (str | List[str], optional): _description_. Defaults to ["*/*.yml"].
            arch_cfg_file (str, optional): _description_. Defaults to "configs/design/architectures/TeMPO_1.yml".
            arch_version (str, optional): _description_. Defaults to "v1".
            input_shape (tuple, optional): _description_. Defaults to (32, 3, 32, 32).
            log_path (str, optional): _description_. Defaults to "log/test_log.txt".
        """
        self.params = Config()
        self.nn_model = nn_model
        self.onn_model = onn_model
        self.params.update(
            {
                "nn_model": (
                    nn_model.__class__.__name__ if nn_model is not None else None
                ),
                "onn_conversion_cfg": onn_conversion_cfg,
                "nn_conversion_cfg": nn_conversion_cfg,
                "onn_model": (
                    onn_model.__class__.__name__ if onn_model is not None else None
                ),
                "model2arch_map_cfg": model2arch_map_cfg,
                "log_path": log_path,
                "devicelib_root": devicelib_root,
                "device_cfg_files": device_cfg_files,
                "arch_cfg_file": arch_cfg_file,
                "arch_version": arch_version,
                "input_shape": input_shape,
            }
        )
        ensure_file_exists(log_path)
        global logger
        logger = get_logger(log_path=log_path)
        self.print_welcome()
        self.initialize()

    def initialize(self) -> None:
        ## Load the device and architecture database
        self.device_db = DeviceLib(
            root=self.params.devicelib_root, config_files=self.params.device_cfg_files
        )

        root, config_file = break_path(self.params.arch_cfg_file)
        self.arch_db = HeteroArchitectureLib(
            root=root, config_file=config_file, version=self.params.arch_version
        )

        self.arch_dim_map = {}

        ## Load the NN model and convert it to ONN model
        if (
            self.params.nn_model is not None
            and self.params.onn_conversion_cfg is not None
            and ONNBaseModel is not None
        ):
            with open(self.params.onn_conversion_cfg, "r") as file:
                map_cfgs = yaml.safe_load(file)
            self.onn_model = ONNBaseModel.from_model(
                self.nn_model, map_cfgs=map_cfgs, verbose=False
            )
        else:
            logger.warning("ONN model not provided")
            self.onn_model = self.nn_model
            try:
                with open(self.params.nn_conversion_cfg, "r") as file:
                    map_cfgs = yaml.safe_load(file)
            except Exception as e:
                raise ValueError("NN conversion config not provided") from e
        if self.onn_model is None:
            err = ValueError("ONN model not provided")
            logger.error(err)
            raise err

        ## Check/verify the layer mapping matches the architecture
        with open(self.params.model2arch_map_cfg, "r") as file:
            model2arch_map_cfg = yaml.safe_load(file)
        # Arch_block: Each corresponding arch blocks for each layer
        # Dim map cfgs: Dimension mapping configurations for each layer
        # layer_to_sub_arch_mapping: Mapping of each layer to the sub-architecture
        # sub_arch_to_layer_mapping: Mapping of each sub-architecture to the layer
        # sub_arch_dictionary: Dictionary of sub-architecture configurations
        (
            self.arch_blocks,
            self.dim_map_cfgs,
            self.layer_to_sub_arch_mapping,
            self.sub_arch_to_layer_mapping,
            self.sub_arch_dictionary,
        ) = check_layer_mapping(self.arch_db.dict(), model2arch_map_cfg, self.onn_model)

        ## Extract the ONN layer-wise workloads
        self.layer_workloads, self.layer_sizes = self.extract_workloads(
            onn_model=self.onn_model,
            input_shape=self.params.input_shape,
            map_cfgs=map_cfgs,
        )

    def extract_workloads(
        self,
        onn_model: nn.Module | None = None,
        input_shape: tuple | None = None,
        map_cfgs: dict | None = None,
    ) -> dict:
        onn_model = onn_model if onn_model is not None else self.onn_model
        input_shape = (
            input_shape if input_shape is not None else self.params.input_shape
        )

        self.layer_info = extract_layer_info(onn_model)
        layer_workloads, layer_sizes = generate_layer_sizes(
            model=onn_model,
            layer_info=self.layer_info,
            input_shape=input_shape,
            map_cfgs=map_cfgs,
        )
        return layer_workloads, layer_sizes

    def simu_partition_cycles(
        self,
        layer_workloads: dict | None = None,
        layer_sizes: dict | None = None,
    ) -> None:
        """
        brief Run the simulation to estimate the number of cycles
        """
        layer_workloads = (
            layer_workloads if layer_workloads is not None else self.layer_workloads
        )

        model_partition_cycles = {}
        for sub_arch_name, layers in self.sub_arch_to_layer_mapping.items():
            model_partition_cycles[sub_arch_name] = {}

            sub_arch_workloads = {layer: layer_workloads[layer] for layer in layers}

            sub_arch_layer_sizes = {layer: layer_sizes[layer] for layer in layers}
            self.arch_dim_map[sub_arch_name] = {}
            for layer_name, matrices in sub_arch_workloads.items():
                (
                    iter_N,
                    iter_D,
                    iter_M,
                    N,
                    D,
                    M,
                    forward_factor_w,
                    forward_factor_x,
                    arch_dims,
                ) = cycles(
                    miniblock=self.arch_blocks[layer_name],
                    matrices=matrices,
                    layer_sizes=sub_arch_layer_sizes[layer_name],
                    dataflow=self.sub_arch_dictionary[sub_arch_name]["dataflow"],
                    op_type=self.sub_arch_dictionary[sub_arch_name]["op_type"],
                    dim_map=self.dim_map_cfgs[layer_name],
                    multi_wavelength=self.sub_arch_dictionary[sub_arch_name]["core"][
                        "num_wavelength"
                    ],
                    work_freq=self.sub_arch_dictionary[sub_arch_name]["core"][
                        "work_freq"
                    ],
                    forward_type=self.sub_arch_dictionary[sub_arch_name]["core"][
                        "forward"
                    ],
                    weight_representation=self.sub_arch_dictionary[sub_arch_name][
                        "core"
                    ]["range"]["weight"],
                    input_representation=self.sub_arch_dictionary[sub_arch_name][
                        "core"
                    ]["range"]["input"],
                    output_representation=self.sub_arch_dictionary[sub_arch_name][
                        "core"
                    ]["range"]["output"],
                )

                model_partition_cycles[sub_arch_name][layer_name] = (
                    iter_N,
                    iter_D,
                    iter_M,
                    N,
                    D,
                    M,
                    forward_factor_w,
                    forward_factor_x,
                )
                self.arch_dim_map[sub_arch_name] = arch_dims

        return model_partition_cycles

    def simu_insertion_loss(self) -> dict:
        sub_arch_insertion_loss = {}
        for sub_arch_name, sub_arch_config in self.sub_arch_dictionary.items():
            sub_arch_insertion_loss[sub_arch_name] = {}
            sub_arch_node_lp, sub_arch_node_il, sub_arch_lp, sub_arch_il = (
                architecture_insertion_loss(
                    sub_arch_config,
                    self.device_db._db,
                )
            )
            sub_arch_insertion_loss[sub_arch_name] = {
                "node_path": sub_arch_node_lp,
                "node_insertion_loss": sub_arch_node_il,
                "core_path": sub_arch_lp,
                "core_insertion_loss": sub_arch_il,
            }
        return sub_arch_insertion_loss

    def simu_memory_cost(
        self,
        model_partition_cycles: dict | None = None,
    ) -> float:
        sub_arch_memory_latency = {}
        sub_arch_memory_energy = {}
        for sub_arch_name, layers in self.sub_arch_to_layer_mapping.items():
            sub_arch_memory_energy[sub_arch_name] = {}
            sub_arch_memory_latency[sub_arch_name] = {}
            sub_arch_layer_sizes = {layer: self.layer_sizes[layer] for layer in layers}
            memory_simulation_results = generate_memory_setting(
                self.sub_arch_dictionary[sub_arch_name],
                model_partition_cycles[sub_arch_name],
                sub_arch_layer_sizes,
                self.arch_dim_map[sub_arch_name],
                self.dim_map_cfgs,
            )

            memory_latency, memory_energy = calculate_memory_latency_and_energy(
                memory_simulation_results,
                model_partition_cycles[sub_arch_name],
                sub_arch_layer_sizes,
                self.layer_to_sub_arch_mapping,
                self.arch_dim_map[sub_arch_name],
                self.dim_map_cfgs,
            )
            sub_arch_memory_latency[sub_arch_name] = memory_latency
            sub_arch_memory_energy[sub_arch_name] = memory_energy
        return (
            sub_arch_memory_latency,
            sub_arch_memory_energy,
            memory_simulation_results,
        )

    def simu_latency(
        self, sub_arch_memory_latency: dict, sub_arch_computation_latency: dict
    ) -> float:
        sub_arch_total_latency = {}
        for sub_arch_name, memory_latency in sub_arch_memory_latency.items():
            sub_arch_total_latency[sub_arch_name] = (
                memory_latency + sub_arch_computation_latency[sub_arch_name]
            )
        return sub_arch_total_latency

    def simu_energy(
        self, model_partition_cycles: dict, insertion_loss_dict: dict
    ) -> float:
        sub_arch_energy_cost = {}
        sub_arch_total_energy = {}
        sub_arch_computation_latency = {}
        for sub_arch_name, layers_workloads in model_partition_cycles.items():
            sub_arch_energy_cost[sub_arch_name] = {}
            sub_arch_total_energy[sub_arch_name] = {}
            sub_arch_computation_latency[sub_arch_name] = {}
            energy_cost, computation_latency = energy_calculator(
                sub_arch=self.sub_arch_dictionary[sub_arch_name],
                device_db=self.device_db._db,
                dim_map_cfgs=self.dim_map_cfgs,
                cycles=layers_workloads,
                insertion_loss=insertion_loss_dict[sub_arch_name][
                    "core_insertion_loss"
                ],
            )
            sub_arch_energy_cost[sub_arch_name] = energy_cost
            sub_arch_total_energy[sub_arch_name] = sum(
                v["total_energy"] for v in energy_cost.values()
            )

            sub_arch_computation_latency[sub_arch_name] = computation_latency

        return sub_arch_energy_cost, sub_arch_total_energy, sub_arch_computation_latency

    def simu_chip_energy(self, energy_breakdown: dict) -> float:
        sub_arch_chip_energy = {}
        for sub_arch_name, energy_cost in energy_breakdown.items():
            sub_arch_chip_energy[sub_arch_name] = chip_energy_calculator(energy_cost)
        return sub_arch_chip_energy

    def simu_chip_area(self, area: dict) -> float:
        sub_arch_chip_area = {}
        for sub_arch_name, area_cost in area.items():
            sub_arch_chip_area[sub_arch_name] = chip_area_calculator(area_cost)
        return sub_arch_chip_area

    def simu_area(self) -> float:
        sub_arch_area_cost = {}
        sub_arch_total_area = {}
        for sub_arch_name, sub_arch_config in self.sub_arch_dictionary.items():
            sub_arch_area_cost[sub_arch_name] = {}
            sub_arch_total_area[sub_arch_name] = {}
            area_cost = area_calculator(
                config=sub_arch_config,
                device_db=self.device_db._db,
            )
            sub_arch_area_cost[sub_arch_name] = area_cost

            total_area = sum(
                v["total_area"] for v in area_cost.values() if "node" not in v
            )
            total_area += area_cost["node"]["total_area"]
            sub_arch_total_area[sub_arch_name] = total_area
        return sub_arch_area_cost, sub_arch_total_area

    def print_welcome(self):
        """
        brief print welcome message
        """
        content = f"""\n\
============================================================================================
                                Simphony: ONN Arch Simulator v{__version__}
                                    Ziang Yin
                        Jiaqi Gu (https://scopex-asu.github.io)"""
        content += "\n================================== Simulation Parameters ===================================\n"
        content += f"\n{self.params}"
        logger.info(content)

    def log_report(self, results, header="Default Report", yaml_style: bool = True):
        content = f"\n================================== Report: {header} ===================================\n"
        if isinstance(results, dict) and yaml_style:
            results_cfg = Config()
            results_cfg.update(results)
            results = results_cfg
        content += f"\n{results}\n"
        logger.info(content)
