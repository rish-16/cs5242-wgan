import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils import save_image

import numpy as np 
from models import *
from utils import *

PATH = "emojis_am/"
train_loader = get_emoji_dataset(PATH)

batch = next(iter(train_loader))
print (batch.shape)

generator = Generator().cuda()
discriminator = Discriminator().cuda()

loss_fn = nn.BCELoss()

# TODO: tune this
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

g_opt = torch.optim.AdamW(generator.parameters(), lr=0.0002)
d_opt = torch.optim.AdamW(discriminator.parameters(), lr=0.0002)
Tensor = torch.cuda.FloatTensor

epochs = 200
for epoch in range(epochs):
    epoch_loss_g = 0
    epoch_loss_d = 0
    n_samples = 0
    for i, imgs in enumerate(train_loader):
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # train generator
        g_opt.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 100)))) # random noise 

        # Generate a batch of images
        gen_imgs = generator(z)
        g_loss = loss_fn(discriminator(gen_imgs), valid) # Loss measures generator's ability to fool the discriminator
        g_loss.backward()
        g_opt.step()

        # train discriminator
        d_opt.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = loss_fn(discriminator(real_imgs), valid)
        fake_loss = loss_fn(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        d_opt.step()

        epoch_loss_d += d_loss.cpu().detach().item()
        epoch_loss_g += g_loss.cpu().detach().item()
        n_samples += 1

    epoch_loss_d = epoch_loss_d / n_samples
    epoch_loss_g = epoch_loss_g / n_samples

    if epoch % 20 == 0:
        print (f"epoch: {epoch} | d_loss: {epoch_loss_d:.5f} | g_loss: {epoch_loss_g:.5f}")
        save_image(gen_imgs.data[:25], f"images/img_run2_b{epoch}.png", nrow=5, normalize=True)
