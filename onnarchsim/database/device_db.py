"""
Date: 2024-08-03 13:22:59
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-08-03 16:30:47
FilePath: /ONNArchSim/onnarchsim/database/device_db.py
"""

import glob
import os
from typing import List

import yaml

from .database import DataBase

__all__ = ["DeviceLib"]


class DeviceLib(DataBase):
    def __init__(
        self, root: str = "configs/devices", config_files: str | List[str] = ["*/*.yml"]
    ) -> None:
        super().__init__()
        self.root = root
        self.config_files = config_files
        self.initialize()

    def initialize(self):
        files = []
        for file in self.config_files:
            files += glob.glob(os.path.join(self.root, file))
        files = set(files)

        for file in files:
            with open(file, "r") as file:
                data = yaml.safe_load(file)  # Parse the YAML file
            self._db.update(data)
