import torch.nn as nn
import torch
from layers import StochasticLinear as SLinear
from layers import StochasticConv2d as SConv2d
from layers import NotStochasticLinear as Linear
from layers import NotStochasticConv2d as Conv2d
from layers import StochasticModel


class Model(StochasticModel):
    def __init__(self, linear=nn.Linear, conv=nn.Conv2d):
        super().__init__()
        self.conv = nn.Sequential(conv(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2),
                                  conv(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=2), 
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2)
                                  )
        self.fc_out = linear(32*7*7, 10)


    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        output = self.fc_out(x)
        return output


class NSModel(Model):
    def __init__(self):
        super().__init__(linear=Linear, conv=Conv2d)


class SModel(Model):
    def __init__(self):
        super().__init__(linear=SLinear, conv=SConv2d)
