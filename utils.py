import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class EmojiDataset(Dataset):
    def __init__(self, img, transform):
        self.img = img
        self.transform = transform

    def __getitem__(self, idx):
        x = self.transform(self.img[idx])
        return x

    def __len__(self):
        return len(self.img)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)        

def get_emoji_dataset(path):
    images = []
    files = os.listdir(path)
    for fle in files:
        # if fle.endswith("png"):
        img = Image.open(path + fle)
        img = np.array(img)
        images.append(img)

    images = np.array(images)

    transform = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = EmojiDataset(images, transform)
    train_loader = DataLoader(dataset, batch_size=64)

    return train_loader