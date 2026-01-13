"""
Date: 2024-01-04 19:46:37
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-01-04 21:27:41
FilePath: /ONNArchSim/onnarchsim/database/device_db.py
"""

import torch
import yaml
from torchonn.models.base_model import ONNBaseModel


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


model = CNN().to("cuda:0")
file_path = "configs/onn_mapping/simple_cnn.yml"

with open(file_path, "r") as file:
    map = yaml.safe_load(file)

## the from_model class method can convert NN to ONN given mapping configs
onn_model = ONNBaseModel.from_model(
    model,
    map_cfgs=map,
    verbose=True,
)

index = 0
for layer in onn_model.named_children():
    print(f"index{index}: {layer}")
    index += 1
