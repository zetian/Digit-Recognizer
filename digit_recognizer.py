import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image


class DigitData(data.Dataset):
    # img_to_tensor = transforms.Compose([transforms.ToTensor()])
    img_to_tensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    def __init__(self, path):
        data = pd.read_csv(path)
        self.images = data.iloc[:, 1:].values
        self.image_size = self.images.shape[1]
        self.image_width = self.image_height = np.ceil(np.sqrt(self.image_size)).astype(np.uint8)
        self.labels = data.iloc[:, 0].values.ravel()
        self.labels_classes = np.unique(self.labels)
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = image.reshape((self.image_width, self.image_height)).astype(np.uint8)
        image = Image.fromarray(image, mode='L')
        image = self.img_to_tensor(image)
        return image, label
    def __len__(self):
        return self.images.shape[0]

    def show_img(self, index):
        image, label = self.images[index], self.labels[index]
        image = image.reshape((self.image_width, self.image_height)).astype(np.uint8)
        plt.imshow(image, cmap='gray')
        plt.title(label)
        plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.data[0]))


train_set = DigitData('input/train.csv')
test_set = DigitData('input/test.csv')
train_set.show_img(1)

# plt.imshow(train_set.__getitem__(0).numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()
# image_1, label_1 = train_set.__getitem__(0)
# print(image_1)
# print("~~~~~")
# print(label_1)

# img_to_tensor = transforms.Compose([transforms.ToTensor()])
# trainset = pd.read_csv('train.csv')
# images = trainset.iloc[:, 1:].values

# image = to_tensor(image)
# train_loader = DataLoader(trainset, batch_size=64,shuffle=True, num_workers=4)
# valset = DigitData('test.csv', transforms)
# val_loader = DataLoader(valset, batch_size=64,shuffle=False, num_workers=4)
# ===========================================
# train_set = pd.read_csv('train.csv')
# images = train_set.iloc[:, 1:].values


# image = images[0]
# image = images[0].reshape((28, 28)).astype(np.uint8)
# image = Image.fromarray(image, mode='L')
# image.show()
# print(image)

# image_size = images.shape[1]
# image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
# image = image.reshape((image_width, image_height)).astype(np.uint8)
# print("~~~~~~~~~~~~~")
# print(image)
# image = image.fromarray(image, mode='L')
# image = to_tensor(image)
# ===========================================

# normalize = transforms.Normalize(
#    mean=[0.485, 0.456, 0.406],
#    std=[0.229, 0.224, 0.225]
# )
# preprocess = transforms.Compose([
#    transforms.Scale(256),
#    transforms.CenterCrop(224),
#    transforms.ToTensor(),
#    normalize
# ])

# img_pil = Image.open("cat.jpg")
# img_tensor = preprocess(img_pil)
# print(img_pil)
# img_tensor.show()
# img_tensor.unsqueeze_(0)

# try_set = pd.read_csv('try.csv')

# images = try_set.iloc[:, 1:].values
# # print(images[0])
# image = images[0].reshape((2, 2)).astype(np.uint8)
# # print(image)
# image = Image.fromarray(image, mode='L')
# # image.show()
# image = img_to_tensor(image)
# print(image)
# image = img_to_tensor(image)
# trainset = pd.read_csv('train.csv')
# train_images = trainset.iloc[:, 1:].values
# train_labels = trainset.iloc[:, 0].values.ravel()

# trainset = pd.read_csv('test.csv')
# test_images = trainset.iloc[:, 1:].values
# test_labels = trainset.iloc[:, 0].values.ravel()

# print(len(test_images))

# print(len(test_labels))