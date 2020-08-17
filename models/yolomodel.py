import os
import sys
import random
import torch
import torchvision
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import resnet_YOLO

model = resnet_YOLO.resnet50()
img = torch.zeros(1, 3, 448, 448)
r = model(img)
print(model)