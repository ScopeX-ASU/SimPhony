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

from .database import DataBase
from .sub_arch_db import SubArchitectureLib
from .utils import break_path, deep_update

__all__ = ["HeteroArchitectureLib"]


class HeteroArchitectureLib(DataBase):
    def __init__(
        self,
        root: str = "configs/design/architectures",
        config_file: str = "TeMPO_hetero.yml",
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

        arch_config = {"hetero_arch": arch_data}

        # for sub_arch in arch_config["hetero_arch"]["sub_archs"]:

        if (
            "sub_archs" in arch_config["hetero_arch"]
            and arch_config["hetero_arch"]["sub_archs"] is not None
        ):
            # print("yes")
            self.load_sub_archs(arch_config["hetero_arch"])

        self._db.update(arch_config)

    def load_sub_archs(self, hetero_arch: Dict[str, Any]) -> None:
        idx = 1
        sub_arch_list = list(hetero_arch["sub_archs"].keys())
        sub_archs_cp = copy.deepcopy(hetero_arch["sub_archs"])
        hetero_arch["sub_archs"] = {}
        print(sub_arch_list)

        for sub_arch_name in sub_arch_list:
            sub_arch_cp = sub_archs_cp[sub_arch_name]
            # print(sub_arch_cp)
            # exit(0)
            if "version" not in sub_arch_cp or sub_arch_cp["version"] == None:
                sub_arch_cp["version"] = "v1"

            if "file" in sub_arch_cp and sub_arch_cp["file"] != "default":
                sub_arch_file_path = sub_arch_cp["file"]
            else:
                sub_arch_file_path = os.path.join(
                    "configs/design/sub_architectures", f"{sub_arch_name}.yml"
                )

            path, arch_file = break_path(sub_arch_file_path)

            sub_arch_total_config_loaded = SubArchitectureLib(
                root=path, config_file=arch_file, version=sub_arch_cp["version"]
            ).dict()

            for name in ["file", "version"]:
                if name in sub_arch_cp:
                    sub_arch_cp.pop(name)

            # print(sub_arch_cp)
            sub_arch_config_loaded = sub_arch_total_config_loaded["sub_arch"]
            deep_update(sub_arch_config_loaded, sub_arch_cp)

            hetero_arch["sub_archs"][f"sub_arch_{idx}"] = sub_arch_config_loaded
            idx += 1
            # print(sub_arch_config_loaded)
            # exit(0)
        # core_cp = copy.deepcopy(core_data["core"][core_name])

        # # core_data = {}
        # # core_config = core_data[core_name]
        # if "version" not in core_cp or core_cp["version"] == None:
        #     core_cp["version"] = "v1"

        # if "file" in core_cp and core_cp["file"] != "default":
        #     core_file_path = core_cp["file"]
        # else:
        #     core_file_path = os.path.join("configs/design/cores", f"{core_name}.yml")

        # path, core_file = break_path(core_file_path)

        # core_total_config_loaded = CoreLib(
        #     root=path, config_file=core_file, version=core_cp["version"]
        # ).dict()
        # for name in ["file", "version"]:
        #     if name in core_cp:
        #         core_cp.pop(name)

        # core_config_loaded = core_total_config_loaded["core"]
        # core_device_loaded = core_total_config_loaded["devices"]

        # # print(core_config_loaded)
        # # # exit(0)
        # # print(core_cp)
        # deep_update(core_config_loaded, core_cp)

        # # print(core_config_loaded)
        # # exit(0)
        # # print(core_data)
        # core_data["core"] = core_config_loaded
        # core_data["core"]["devices"] = core_device_loaded
