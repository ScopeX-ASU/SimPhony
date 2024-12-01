"""
Date: 2024-08-03 13:22:59
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-08-03 14:15:46
FilePath: /ONNArchSim/onnarchsim/database/utils.py
"""

import os
from typing import Tuple

__all__ = [
    "ensure_file_exists",
    "deep_update",
    "break_path",
]


def ensure_file_exists(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Create the file if it does not exist
    with open(file_path, "w") as f:
        pass


def break_path(
    file_path: str = "configs/design/architectures/TeMPO.yml",
) -> Tuple[str, str]:
    directory, filename = os.path.split(file_path)
    return directory, filename


def deep_update(dict1, dict2):
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            deep_update(dict1[key], value)
        else:
            dict1[key] = value
