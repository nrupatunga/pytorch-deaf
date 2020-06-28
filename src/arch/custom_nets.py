"""
File: custom_net.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: custom network architecture
"""
import torch.nn as nn
from torchsummary import summary


class EdgeAwareNet(nn.Module):

    def __init__(self, multiplier=1):

        super().__init__()

        print('Arch Multiplier: {}'.format(multiplier))

        # TODO: do it in a better way
        nbLayers0 = 512
        nbLayers1 = 512

        self.conv1 = nn.Conv2d(3, nbLayers0, kernel_size=16,
                               stride=1, padding=3, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(nbLayers0, nbLayers1, kernel_size=1,
                               stride=1, padding=2, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(nbLayers1, 3, kernel_size=8,
                               stride=1, padding=2, bias=False)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x_in):

        # features
        x = self.conv1(x_in)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        return x


if __name__ == "__main__":
    net = EdgeAwareNet()
    summary(net, input_size=(3, 64, 64))
