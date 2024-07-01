from torch import nn
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)
