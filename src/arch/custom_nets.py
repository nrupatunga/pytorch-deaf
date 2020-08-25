"""
File: custom_net.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: custom network architecture
"""
import torch.nn as nn
from torchsummary import summary
import torch.nn.init as init


class EdgeAwareNet(nn.Module):

    def __init__(self, multiplier=1):

        super().__init__()

        print('Arch Multiplier: {}'.format(multiplier))

        # TODO: do it in a better way
        nbLayers0 = 256
        nbLayers1 = 256
        self.layers = []

        self.conv1 = nn.Conv2d(3, nbLayers0, kernel_size=16,
                               stride=1, padding=4, bias=True)
        # self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin1 = nn.Tanh()

        self.conv2 = nn.Conv2d(nbLayers0, nbLayers1, kernel_size=1,
                               stride=1, padding=0, bias=True)
        # self.nonlin2 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.Tanh()

        self.conv3 = nn.Conv2d(nbLayers1, 3, kernel_size=8,
                               stride=1, padding=3, bias=True)
        self.nonlin3 = nn.Tanh()

        self.layers.append(self.conv1)
        self.layers.append(self.nonlin1)
        self.layers.append(self.conv2)
        self.layers.append(self.nonlin2)
        self.layers.append(self.conv3)
        self.init_weights()

    def forward(self, x_in):

        # features
        x = self.conv1(x_in)
        x = self.nonlin1(x)

        x = self.conv2(x)
        x = self.nonlin2(x)

        x = self.conv3(x)
        x = self.nonlin3(x)

        return x

    def init_weights(self):
        """Initialize the extra layers """
        for m in self.layers:
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight.data, mean=0.0, std=1e-3)
                init.zeros_(m.bias.data)


if __name__ == "__main__":
    net = EdgeAwareNet().cuda()
    summary(net, input_size=(3, 270, 270))
