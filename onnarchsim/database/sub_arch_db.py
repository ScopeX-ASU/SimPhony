"""
Date: 2024-01-04 19:46:37
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-01-04 21:27:41
FilePath: /ONNArchSim/onnarchsim/database/device_db.py
"""

import copy
import os
from typing import Any, Dict

import yaml

from .core_db import CoreLib
from .database import DataBase
from .utils import break_path, deep_update

__all__ = ["SubArchitectureLib"]


class SubArchitectureLib(DataBase):
    def __init__(
        self,
        root: str = "configs/design/sub_architectures",
        config_file: str = "TeMPO_1.yml",
        version: str = "v1",
    ) -> None:
        super().__init__()
        self.root = root
        self.config_file = config_file
        self.version = version
        self.initialize()

    def initialize(self):
        file_path = os.path.join(self.root, self.config_file)
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        arch_name = list(data.keys())[0]
        arch_versions = data[arch_name]

        if self.version not in arch_versions:
            raise ValueError(
                f"Version '{self.version}' not found in the architecture config."
            )

        arch_data = arch_versions[self.version]
        arch_data["version"] = self.version
        arch_data["name"] = arch_name

        arch_config = {"sub_arch": arch_data}

        if (
            "core" in arch_config["sub_arch"]
            and arch_config["sub_arch"]["core"] is not None
        ):
            self.load_core(arch_config["sub_arch"])
            # print(arch_config)

        # if (
        #     "sub_archs" in arch_config["architecture"]
        #     and arch_config["architecture"]["sub_archs"] is not None
        # ):
        #     # print("yes")
        #     self.load_sub_arch(arch_config)

        self._db.update(arch_config)

        # print("Here: ", self.arch_config)

    # def load_sub_arch(self, arch_data: Dict[str, Any]) -> None:
    #     sub_archs = copy.deepcopy(arch_data["architecture"]["sub_archs"])
    #     # sub_archs = main_arch['architecture'].get('sub_archs', {})
    #     arch_data["architecture"]["sub_archs"] = {}

    #     # print(updated_sub_archs)

    #     index = 1
    #     for sub_arch_name, sub_arch_config in sub_archs.items():
    #         # print("From main arch: ", sub_arch_config)
    #         if "version" not in sub_arch_config or sub_arch_config["version"] == None:
    #             sub_arch_config["version"] = "v1"

    #         if "file" in sub_arch_config and sub_arch_config["file"] != "default":
    #             sub_arch_file_path = sub_arch_config["file"]
    #         else:
    #             sub_arch_file_path = os.path.join(self.root, f"{sub_arch_name}.yml")

    #         path, arch_file = break_path(sub_arch_file_path)

    #         sub_arch_config_loaded = ArchitectureLib(
    #             root=path, config_file=arch_file, version=sub_arch_config["version"]
    #         ).dict()["architecture"]
    #         # print(sub_arch_config_loaded)
    #         # print("Before update arch: ", sub_arch_config_loaded)
    #         for name in ["file", "version"]:
    #             if name in sub_arch_config:
    #                 sub_arch_config.pop(name)
    #         deep_update(sub_arch_config_loaded, sub_arch_config)
    #         # print("After update arch: ", sub_arch_config_loaded)

    #         arch_data["architecture"]["sub_archs"][f"sub_arch{index}"] = (
    #             sub_arch_config_loaded
    #         )

    #         # print(updated_sub_archs)

    #         index += 1

    #     return
    # arch_data['architecture']['sub_archs'][f"sub_arch{index}"] = {}
    # arch_data['architecture']['sub_archs'][f"sub_arch{index}"].update(sub_arch_loaded_config)
    # print(sub_arch_loaded_config)
    # exit(0)

    def load_core(self, core_data: Dict[str, Any]) -> None:
        core_name = list(core_data["core"].keys())[0]
        core_cp = copy.deepcopy(core_data["core"][core_name])

        # core_data = {}
        # core_config = core_data[core_name]
        if "version" not in core_cp or core_cp["version"] == None:
            core_cp["version"] = "v1"

        if "file" in core_cp and core_cp["file"] != "default":
            core_file_path = core_cp["file"]
        else:
            core_file_path = os.path.join("configs/design/cores", f"{core_name}.yml")

        path, core_file = break_path(core_file_path)

        core_total_config_loaded = CoreLib(
            root=path, config_file=core_file, version=core_cp["version"]
        ).dict()
        for name in ["file", "version"]:
            if name in core_cp:
                core_cp.pop(name)

        core_config_loaded = core_total_config_loaded["core"]
        core_device_loaded = core_total_config_loaded["devices"]

        # print(core_config_loaded)
        # # exit(0)
        # print(core_cp)
        deep_update(core_config_loaded, core_cp)

        # print(core_config_loaded)
        # exit(0)
        # print(core_data)
        core_data["core"] = core_config_loaded
        core_data["core"]["devices"] = core_device_loaded
