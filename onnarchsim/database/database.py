"""
Date: 2024-01-04 19:46:37
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-01-04 19:59:07
FilePath: /ONNArchSim/src/database.py
"""

from typing import Any, Dict, List, Tuple, Union

from multimethod import multimethod
from pyutils.config import Config

__all__ = ["DataBase"]


class DataBase(object):
    def __init__(self) -> None:
        self._db = Config()

    def load(self, fpath: str, *, recursive: bool = False) -> None:
        self._db.load(fpath, recursive=recursive)

    def reload(self, fpath: str, *, recursive: bool = False) -> None:
        self._db.reload(fpath, recursive=recursive)

    @multimethod
    def update(self, other: Dict) -> None:
        self._db.update(other)

    @multimethod
    def update(self, opts: Union[List, Tuple]) -> None:
        self._db.update(opts)

    def dict(self) -> Dict[str, Any]:
        return self._db.dict()

    def flat_dict(self) -> Dict[str, Any]:
        return self._db.flat_dict()

    def hash(self) -> str:
        return self._db.hash()

    def dump_to_yml(self, path: str) -> None:
        self._db.dump_to_yml(path)

    def __str__(self) -> str:
        return str(self._db)
