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
    def __init__(self, in_channels, out_channels=1):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128 * 64 * 64),
            nn.ReLU(True),
        )

        self.conv = nn.Sequential(
            BlockDecoder(128, 64),
            BlockDecoder(64, 32),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 64, 64)
        x = self.conv(x)
        return x
