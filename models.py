import os
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                        nn.Conv2d(3, 16, (3,3)),
                        nn.MaxPool2d((2,2)),
                        nn.ReLU(),
                        nn.Conv2d(16, 64, (3,3)),
                        nn.MaxPool2d((2,2)),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, (3,3)),
                        nn.MaxPool2d((2,2)),
                        nn.ReLU()
                    )

        self.mlp = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(25088, 1024),
                        nn.SiLU(),
                        nn.Linear(1024, 256),
                        nn.SiLU(),
                        nn.Linear(256, 64),
                        nn.SiLU(),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                    )

    def forward(self, x):
        # x is a 128x128x3 image
        x = self.net(x)
        out = self.mlp(x)

        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 128 // 4
        self.l1 = nn.Sequential(
                        nn.Linear(100, 128 * self.init_size ** 2) # 100 is the latest noise dimension
                    )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# x = torch.rand(5, 3, 128, 128)
# net = Discriminator()
# y = net(x) # [5, 1]