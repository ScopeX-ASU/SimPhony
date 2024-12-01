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
from .node_db import NodeLib
from .utils import break_path, deep_update

__all__ = ["CoreLib"]


class CoreLib(DataBase):
    def __init__(
        self,
        root: str = "configs/design/cores",
        config_file: str = "dptc.yml",
        version: str = "v1",
    ) -> None:
        super().__init__()
        self.root = root
        self.config_file = config_file.lower()
        self.version = version
        self.initialize()

    def initialize(self):
        file_path = os.path.join(self.root, self.config_file)
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        core_name = list(data.keys())[0]
        core_versions = data[core_name]

        if self.version not in core_versions:
            raise ValueError(f"Version '{self.version}' not found in the core config.")

        core_data = core_versions[self.version]
        core_data["core"]["version"] = self.version
        core_data["core"]["name"] = core_name
        #         # print(processed_cores)

        if "node" in core_data["core"] and core_data["core"] is not None:
            self.load_node(core_data["core"])

        #         # self.load_nodes(cores_config=self.processed_config [f"core{index}"]['nodes'])

        #         index += 1
        #     # print("Here: ", processed_cores)
        self._db.update(core_data)
        # print(self._db)
        # exit(0)
        # print(self._db)
        # exit(0)

    def load_node(self, node_data: Dict[str, Any]) -> None:
        node_name = list(node_data["node"].keys())[0]
        node_cp = copy.deepcopy(node_data["node"][node_name])

        # core_data = {}
        # core_config = core_data[core_name]
        if "version" not in node_cp or node_cp["version"] == None:
            node_cp["version"] = "v1"

        if "file" in node_cp and node_cp["file"] != "default":
            node_file_path = node_cp["file"]
        else:
            node_file_path = os.path.join("configs/design/nodes", f"{node_name}.yml")

        path, node_file = break_path(node_file_path)

        node_total_config_loaded = NodeLib(
            root=path, config_file=node_file, version=node_cp["version"]
        ).dict()
        for name in ["file", "version"]:
            if name in node_cp:
                node_cp.pop(name)

        node_config_loaded = node_total_config_loaded["node"]
        node_device_loaded = node_total_config_loaded["devices"]

        # print(core_config_loaded)
        # # exit(0)
        # print(core_cp)
        deep_update(node_config_loaded, node_cp)

        # print(core_config_loaded)
        # exit(0)
        # print(core_data)
        node_data["node"] = node_config_loaded
        node_data["node"]["devices"] = node_device_loaded
