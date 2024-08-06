import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import os

from CAE import CAE
from Encoder import Encoder
from Decoder import Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameter setting

in_channels = 1
out_channels = 1
latent_dim = 10
num_epochs = 20
batch_size = 128
learning_rate = 5e-4
b1 = 0.95
b2 = 0.99

# Loading the data

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

mnist_train = datasets.MNIST(root='./mnist_images', train=True, download=True, transform=transform)
mnist_dataset = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

if not os.path.exists('output_dir'):
    os.makedirs('output_dir')

encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
decoder = Decoder(in_channels=latent_dim, out_channels=out_channels)
model = CAE(encoder, decoder)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(b1, b2))

for e in range(num_epochs):
    for i, (images, _) in enumerate(mnist_dataset):
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epoch [{}/{}], batch [{}/{}], loss:{:.4f}'
              .format(e + 1, num_epochs, i + 1, len(mnist_dataset), loss.item()))

        if i == len(mnist_dataset) - 1:
            pic = outputs.cpu().data
            save_image(pic, './output_dir/image_{}.png'.format(e + 1))
