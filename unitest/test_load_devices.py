import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from onnarchsim.database.device_db import DeviceLib
from onnarchsim.database.hetero_arch_db import HeteroArchitectureLib
from onnarchsim.workflow.utils import load_required_devices

# Used for checking whether the loaded config is correct


def test_sub_arch_db():
    hetero_arch = HeteroArchitectureLib(config_file="LT_hetero.yml").dict()
    sub_arch = hetero_arch["hetero_arch"].get("sub_archs", {}).get("sub_arch_1", {})

    device_db = DeviceLib().dict()

    components, operand_1, operand_2, temporal_shared = load_required_devices(
        sub_arch, device_db
    )

    for key, value in components.items():
        print(key, value)
        print("\n")
    print("\n")
    for key in operand_1:
        print(key)
        print("\n")
    print("\n")
    for key in operand_2:
        print(key)
        print("\n")
    print("\n")
    for key, value in temporal_shared.items():
        print(key, value)
        print("\n")
    print("\n")


test_sub_arch_db()
