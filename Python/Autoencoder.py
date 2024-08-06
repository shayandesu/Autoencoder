import torch
import torch.nn as nn
from Decoder import Decoder
from Encoder import Encoder


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(AutoEncoder, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = Encoder(latent_dim).to(device)
        self.decoder = Decoder(latent_dim).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
