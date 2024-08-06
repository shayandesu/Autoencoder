import torch.nn as nn


class BlockEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            BlockEncoder(in_channels=in_channels, out_channels=32),
            BlockEncoder(in_channels=32, out_channels=64),
            BlockEncoder(in_channels=64, out_channels=64),
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=latent_dim)
        )

    def forward(self, x):
        return self.model(x)