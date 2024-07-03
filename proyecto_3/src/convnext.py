from torch import nn
from torchvision import models


class ConvNext(nn.Module):
    def __init__(self, num_classes):
        super(ConvNext, self).__init__()

        # Load a pre-trained ConvNeXt model
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

        # Replace the classifier layer
        num_ftrs = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.convnext(x)
