"""
Date: 2024-01-04 19:46:37
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-01-04 21:27:41
FilePath: /ONNArchSim/onnarchsim/database/device_db.py
"""

import os

import yaml

from .database import DataBase

__all__ = ["NodeLib"]


class NodeLib(DataBase):
    def __init__(
        self,
        root: str = "configs/design/nodes",
        config_file: str = "dode.yml",
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

        node_name = list(data.keys())[0]
        node_versions = data[node_name]

        if self.version not in node_versions:
            raise ValueError(f"Version '{self.version}' not found in the node config.")

        node_data = node_versions[self.version]
        node_data["node"]["version"] = self.version
        node_data["node"]["name"] = node_name
        # print(node_data)
        #     files = []
        #     for file in self.config_files:
        #         files += glob.glob(os.path.join(self.root, file))
        #     files = set(files)
        #     # print(files)

        #     for file in files:
        #         with open(file, "r") as file:
        #             data = yaml.safe_load(file)  # Parse the YAML file
        #         for name, core in data.items():
        #             # print(core)
        #             for name, version in core.items():
        #                 # print(version)
        #                 # exit(0)
        #                 self.resolve_references(version)
        self._db.update(node_data)
        # print(self._db)
        # exit(0)

    # def resolve_references(self, config: Dict[str, Any]) -> None:
    #     """Resolve device references in the node configuration."""
    #     resolved_devices = {}
    #     for device_type, device_names in config.get('devices', {}).items():
    #         if device_type not in self.device_lib._db:
    #             raise ValueError(f"Device type '{device_type}' not found in device library.")
    #         for i, name in enumerate(device_names):
    #             if name in self.device_lib._db[device_type]:
    #                 resolved_devices[f"{device_type}[{i}]"] = self.device_lib._db[device_type][name]
    #             else:
    #                 raise ValueError(f"Device reference '{name}' not found in device library.")

    #     instances = config.get('node', {}).get('netlist', {}).get('instances', {})
    #     for instance, ref in instances.items():
    #         if ref in resolved_devices:
    #             # Deepcope
    #             instances[instance] = resolved_devices[ref]
    #         else:
    #             raise ValueError(f"Instance reference '{ref}' not found in device definitions.")
