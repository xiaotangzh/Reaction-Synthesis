from torch import nn
import torch

class GAN(nn.Module):
    def __init__(self, in_channels):
        super(GAN, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.squeeze()

        return x

