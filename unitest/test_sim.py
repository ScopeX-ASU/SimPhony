"""
Date: 2024-08-05 14:20:56
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-08-05 23:02:11
FilePath: /ONNArchSim/unitest/test_sim.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.quantized_BERT_base import QBertViTBase
from onnarchsim.simulator import ONNArchSimulator

config = {
    "conv_cfg": {
        "type": "QConv2d",
        "w_bit": 4,
        "in_bit": 4,
        "out_bit": 4,
    },
    "linear_cfg": {
        "type": "QLinear",
        "w_bit": 4,
        "in_bit": 4,
        "out_bit": 4,
    },
    "matmul_cfg": {
        "type": "QMatMul",
        "w_bit": 4,
        "in_bit": 4,
        "out_bit": 4,
    },
    "norm_cfg": {
        "type": "BN2d",
        "affine": True,
    },
    "act_cfg": {
        "type": "ReLU6",
        "inplace": True,
    },
}


def test_sim():
    model = QBertViTBase(
        conv_cfg=config["conv_cfg"],
        linear_cfg=config["linear_cfg"],
        matmul_cfg=config["matmul_cfg"],
    ).to("cuda:0")

    sim = ONNArchSimulator(
        nn_model=model,
        onn_conversion_cfg=None,
        nn_conversion_cfg="configs/nn_mapping/simple_cnn.yml",
        onn_model=None,
        model2arch_map_cfg="configs/architecture_mapping/simple_lt_cnn.yml",
        devicelib_root="configs/devices",
        device_cfg_files=["*/*.yml"],
        arch_cfg_file="configs/design/architectures/LT_hetero.yml",
        arch_version="v1",
        input_shape=(2, 3, 224, 224),
        log_path="log/test_lt_log.txt",
    )
    partition_cycles = sim.simu_partition_cycles(sim.layer_workloads, sim.layer_sizes)

    sim.log_report(partition_cycles, header="Partition Cycles")

    insertion_loss_breakdown = sim.simu_insertion_loss()
    sim.log_report(insertion_loss_breakdown, header="Insertion Loss (Breakdown)")
    energy_cost_breakdown, total_energy, final_computation_latency = sim.simu_energy(
        partition_cycles, insertion_loss_breakdown
    )
    chip_energy_breakdown = sim.simu_chip_energy(energy_cost_breakdown)
    area_cost_breakdown, total_area_cost = sim.simu_area()
    chip_area_breakdown = sim.simu_chip_area(area_cost_breakdown)
    memory_latency, memory_energy, memory_spec = sim.simu_memory_cost(partition_cycles)

    sim.log_report(energy_cost_breakdown, header="Energy Cost (Breakdown) (pJ)")
    sim.log_report(total_energy, header="Total Energy (pJ)")
    sim.log_report(final_computation_latency, header="Final Computation Latency (s)")
    sim.log_report(chip_energy_breakdown, header="Chip Energy (pJ)")
    sim.log_report(area_cost_breakdown, header="Area Cost (Breakdown) (um^2)")
    sim.log_report(total_area_cost, header="Total Area (um^2)")
    sim.log_report(chip_area_breakdown, header="Chip Area (um^2)")
    sim.log_report(memory_latency, header="Memory Latency (s)")
    sim.log_report(memory_energy, header="Memory Energy (pJ)")
    sim.log_report(memory_spec, header="Memory Specification")


if __name__ == "__main__":
    test_sim()
