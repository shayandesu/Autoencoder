import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import os

from Autoencoder import AutoEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameter setting

latent_dim = 10
num_epochs = 20
batch_size = 128
learning_rate = 5e-4
b1 = 0.95
b2 = 0.99

# Loading the data

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_train = datasets.MNIST(root='./mnist_images', train=True, download=True, transform=transform)
mnist_dataset = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

if not os.path.exists('output_dir'):
    os.makedirs('output_dir')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


autoencoder = AutoEncoder(latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, betas=(b1, b2))
autoencoder.train()

for e in range(num_epochs):
    for i, (images, _) in enumerate(mnist_dataset):
        images = images.to(device)
        outputs = autoencoder(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'
              .format(e + 1, num_epochs, loss.item()))

        if (e + 1) % 5 == 0 and i == len(mnist_dataset) - 1:
            pic = to_img(outputs.cpu().data)
            save_image(pic, './output_dir/image_{}.png'.format(e + 1))
