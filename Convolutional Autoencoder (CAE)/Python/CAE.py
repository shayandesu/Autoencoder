import torch.nn as nn

class CAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(CAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
