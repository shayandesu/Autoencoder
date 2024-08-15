import torch.nn as nn
from Decoder import Decoder
from Encoder import Encoder


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, z_dim, output_size, random_seed):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, z_dim, random_seed)
        self.decoder = Decoder(z_dim, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
