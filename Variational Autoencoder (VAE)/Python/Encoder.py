import torch
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


class Sample(nn.Module):
    def __init__(self, seed):
        super(Sample, self).__init__()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def forward(self, z_mean, z_log_var):
        eps = torch.randn_like(z_log_var)
        z = z_mean + torch.exp(0.5 * z_log_var) * eps
        return z


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim, random_seed):
        super(Encoder, self).__init__()
        self.model - nn.Sequential(
            BlockEncoder(in_channels, 32),
            BlockEncoder(32, 64),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, out_channels)
        )

        self.zm = nn.Linear(out_channels, z_dim)
        self.zlv = nn.Linear(out_channels, z_dim)
        self.sample = Sample(random_seed)

    def forward(self, x):
        x = self.model(x)
        z_mean = self.zm(x)
        z_log_var = self.zlv(x)
        z = self.sample(z_mean, z_log_var)
        return z, z_mean, z_log_var
