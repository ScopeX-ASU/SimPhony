import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from onnarchsim.simulator import ONNArchSimulator


# Define any model here
# Or you can directly get the model from pre-defined models from our models folder
class CNN(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(3, 8, 3, bias=False)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, bias=False)
        self.pool = torch.nn.AdaptiveAvgPool2d((5, 5))
        self.linear = torch.nn.Linear(200, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


# If you want to convert model using TorchONN
# Please provide the file
onn_conversion_cfg = "configs/onn_mapping/simple_cnn.yml"

# Otherwise, we can work with unconverted model
# But loses the ability to train the model from analog side
nn_conversion_cfg = "configs/nn_mapping/simple_cnn.yml"

# This configuration is requried for simulator to properly map the model to the architecture
model2arch_map_cfg = "configs/architecture_mapping/simple_cnn.yml"

model = CNN()

sim = ONNArchSimulator(
    nn_model=model,
    onn_conversion_cfg=onn_conversion_cfg,
    nn_conversion_cfg=nn_conversion_cfg,
    onn_model=None,
    model2arch_map_cfg=model2arch_map_cfg,
    devicelib_root="configs/devices",
    device_cfg_files=["*/*.yml"],
    arch_cfg_file="configs/design/architectures/LT_hetero.yml",
    arch_version="v1",
    input_shape=(2, 3, 32, 32),
    log_path="log/test_memory.txt",
)

# After the partitioned cycles been calculated
partition_cycles = sim.simu_partition_cycles(sim.layer_workloads, sim.layer_sizes)

# We can generate the memory setting and calculate the memory cost
memory_latency, memory_energy, memory_spec = sim.simu_memory_cost(partition_cycles)
sim.log_report(memory_latency, header="Memory Latency (s)")
sim.log_report(memory_energy, header="Memory Energy (pJ)")
sim.log_report(memory_spec, header="Memory Specification")