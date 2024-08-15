import torch.nn as nn


class BlockDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockDecoder, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128 * 64 * 64),
            nn.ReLU(inplace=True)
        )
        self.dec = nn.Sequential(
            BlockDecoder(in_channels=128, out_channels=64),
            BlockDecoder(in_channels=64, out_channels=16),
            nn.ConvTranspose2d(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.dec(x)
        return x