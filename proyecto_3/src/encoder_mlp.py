import torch
import torch.nn as nn


class EncoderMLP(nn.Module):
    def __init__(self, encoder, mlp):
        super(EncoderMLP, self).__init__()
        self.encoder = encoder
        self.mlp = mlp

    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.mlp(x)
        return x

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
