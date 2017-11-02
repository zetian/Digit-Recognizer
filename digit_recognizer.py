import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# class DigitData(data.Dataset):
#     def __init__(self, path, transform):
#         self.transform = transform
#         data = pd.read_csv(path)
#         self.images = data.iloc[:, 1:].values
#         self.image_size = self.images.shape[1]
#         self.image_width = self.image_height = np.ceil(np.sqrt(self.image_size)).astype(np.uint8)
#         self.labels = data.iloc[:, 0].values.ravel()
#         self.labels_classes = np.unique(self.labels)
#     def getitem(self, index):
#         image, label = self.images[index], self.labels[index]
#         image = image.reshape((self.image_width, self.image_height)).astype(np.uint8)
#         # image = image.fromarray(image, mode='L')
#         image = self.transform(image)
#         return image, label
#     def len(self):
#         return self.images.shape[0]

# transforms = transforms.Compose([transforms.ToTensor()])
# trainset = DigitData('train.csv', transforms)
# train_loader = DataLoader(trainset, batch_size=64,shuffle=True, num_workers=4)
# valset = DigitData('test.csv', transforms)
# val_loader = DataLoader(valset, batch_size=64,shuffle=False, num_workers=4)

trainset = pd.read_csv('train.csv')
train_images = trainset.iloc[:, 1:].values
train_labels = trainset.iloc[:, 0].values.ravel()

print(len(train_images))

print(len(train_labels))