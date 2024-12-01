import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from onnarchsim.database.hetero_arch_db import HeteroArchitectureLib

sys.path.pop(0)


def test_db():
    root = "configs/design/architectures"
    config_file = "LT_hetero.yml"
    db = HeteroArchitectureLib(root=root, config_file=config_file, version="v1")
    print(db)

test_db()
