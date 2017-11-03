import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
from PIL import Image

# Hyper Parameters
EPOCH = 20        
BATCH_SIZE = 50
LR = 0.001          


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


class TestData(data.Dataset):
    # img_to_tensor = transforms.Compose([transforms.ToTensor()])
    img_to_tensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    def __init__(self, path):
        data = pd.read_csv(path)
        self.images = data.iloc[:, :].values
        self.image_size = self.images.shape[1]
        self.image_width = self.image_height = np.ceil(np.sqrt(self.image_size)).astype(np.uint8)
    def __getitem__(self, index):
        image = self.images[index]
        image = image.reshape((self.image_width, self.image_height)).astype(np.uint8)
        image = Image.fromarray(image, mode='L')
        image = self.img_to_tensor(image)
        return image
    def __len__(self):
        return self.images.shape[0]



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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


# optimizer = optim.SGD(cnn_net.parameters())

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
train_set = DataLoader(train_set, batch_size = 64, shuffle = True, num_workers = 4)
# print(train_set)
test_set = TestData('input/test.csv')
test_set = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers = 4)
# train_set.show_img(1)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = 64, shuffle = True)


cnn_net = CNN()
# print(cnn_net)
optimizer = torch.optim.Adam(cnn_net.parameters(), lr = LR)
loss_func = nn.CrossEntropyLoss()


def train(epoch):
    cnn_net.train()
    for batch_idx, (data, target) in enumerate(train_set):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = cnn_net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_set.dataset),
                100. * batch_idx / len(train_set), loss.data[0]))

def pred():
    cnn_net.eval()
    pred = []
    test_loss = 0
    correct = 0

    for data in test_set:
        data = Variable(data, volatile=True)
        output = cnn_net(data)
        pred_single = output.data.max(1, keepdim=True)[1][0][0] # get the index of the max log-probability
        # print(pred_single)
        pred.append(pred_single)
        # break
    return pred




def test():
    cnn_net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = cnn_net(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# for epoch in range(1, EPOCH + 1):
#     train(epoch)
#     test()
cnn_net.load_state_dict(torch.load('mnist_training.pt'))
image_id = []
for i in range(1, len(test_set) + 1):
    image_id.append(i)
# image_id = [range(1, 10 + 1)]
# print(image_id)
test()
prediction = pred()

# print(prediction)


raw_data = {'ImageId': image_id,
        'Label': prediction}
df = pd.DataFrame(raw_data, columns = ['ImageId', 'Label'])
df.to_csv('submisiion.csv', index=False)

# raw_data = {'ImageId': [1,2,3], 'Label': [1,2,3]}
# df = pd.DataFrame(raw_data, columns = ['ImageId', 'Label'])
# df.to_scv('submisiion.csv')
# submission.to_csv('submisiion.csv', sep='\t')
# print(df)

# submission = [['ImageId', 'Label']]
# submission.append([1,2])
# df = pd.DataFrame(data = submission)
# print(df)
# test()
# torch.save(cnn_net.state_dict(), 'mnist_training.pt')
