from torchvision import models
import torch.nn as nn

def build_densenet():
    model = models.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(1024, 10)  # 10 classes for MNIST and FMNIST
    return model
