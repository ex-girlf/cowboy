# 搭建网络模型
import torch
from torch.nn import Module, Conv2d, Sequential, MaxPool2d, Flatten, Linear


class Wmodel(Module):
    def __init__(self):
        super(Wmodel, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x= self.model(x)
        return x


if __name__ == '__main__':
    m = Wmodel()
    input = torch.ones((64, 3, 32, 32))
    output = m(input)
    print(output)