#  coding=utf-8
import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


def prf(next):
    next = np.array(next, dtype='uint32')
    next = next * np.array(1103515245, dtype='uint32') + np.array(12345, dtype='uint32')
    return int((next//65536) % 60000)

def save_conv(conv, path="par.txt"):
    ii = conv.weight.shape[0]
    jj = conv.weight.shape[1]
    with open(path, "wb") as f:
        for i in range(ii):
            for j in range(jj):
                np.savetxt(f, conv.weight[i][j].cpu().detach().numpy(), fmt=fmtt)
        np.savetxt(f, conv.bias.cpu().detach().numpy().reshape((1, -1)), fmt=fmtt)


def save_fc(fc, path="fc.txt"):
    with open(path, "wb") as f:
        np.savetxt(f, fc.weight.cpu().detach().numpy(), fmt=fmtt)
        np.savetxt(f, fc.bias.cpu().detach().numpy().reshape((1, -1)), fmt=fmtt)


def save_net(net, base_path):
    folder = os.path.exists(base_path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(base_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

    save_conv(net.conv1, base_path + "/conv1.txt")
    save_conv(net.conv2, base_path + "/conv2.txt")
    save_fc(net.fc1, base_path + "/fc1.txt")
    save_fc(net.fc2, base_path + "/fc2.txt")
    save_fc(net.fc3, base_path + "/fc3.txt")

def init_checkpoint(path):
    if os.path.exists(path):
        if os.listdir(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
    else:
        os.makedirs(path)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    num_work = 0
    batch_size = 128
    ckpt_base_path = "./Parameter/"
    with open("seed", "r") as f:
        seed = f.read()
        seed = int(seed)
    # max_iter = 460s
    max_iter = int(sys.argv[1])
    print("[Untrusted]: start training with device:{}, seed:{}".format(device, seed));

    # print('device:', device)
    # print('seed:', seed)
    transform = transforms.Compose(
        [transforms.ToTensor(), ])

    trainset = torchvision.datasets.MNIST(root=r'./data', train=True,
                                          download=True, transform=transform)

    testset = torchvision.datasets.MNIST(root=r'./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_work)


    fmtt = "%.8lf"




    net = Net()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    start = time.clock()
    #####不进行优化的准确率
    correct = 0
    total = 0
    for data in testloader:
        image, labels = data[0].to(device), data[1].to(device)
        image = image * 255
        outputs = net(image)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()
    print('[Untrusted]: Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    #####
    print("[Untrusted]: Training...")
    init_checkpoint(ckpt_base_path)
    save_net(net, os.path.join(ckpt_base_path, str(0)))
    for i in tqdm(range(max_iter)):
        inputs = torch.zeros((batch_size, 1, 28, 28))
        labels = torch.LongTensor(batch_size)
        for j in range(batch_size):
            index = prf(seed * i + j)
            labels[j] = trainset[index][1]
            inputs[j] = trainset[index][0]
        # get the inputs; data is a list of [inputs, labels]
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs * 255
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        save_net(net, os.path.join(ckpt_base_path, str(i+1)))
        # print("Untrusted: round:"+ str(i + 1))


    correct = 0
    total = 0
    for data in testloader:
        image, labels = data[0].to(device), data[1].to(device)
        image = image * 255
        outputs = net(image)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('[Untrusted]: Accuracy of the network on the 10000 test images: %d %%\n\n' % (
            100 * correct / total))

    print('[Untrusted]: Finished Training')
    elapsed = (time.clock() - start)
    print("[Untrusted]: Time used:", elapsed)

