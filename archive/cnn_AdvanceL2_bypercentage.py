import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import csv

from utils import *

test_name = "cnn_AdvanceL2_by5pctg_wobn_200epochs"
netid = "pwf227"
# Hyper-parameters 
num_epochs = 200 #6
batch_size = 128 #128
learning_rate = 0.01
batch_norm = False

Lambda_L2s = []
epochs, steps = [], []
train_bareloss, train_loss, train_acc = [], [], []
test_bareloss, test_loss, test_acc = [], [], []

use_AutoL2 = True
k = 5 # make measurements every k steps
min_step = 0
decay_factor_L2 = 0.1

# if min_loss or accuracy increase by this percentage
# then update L2 lambda
increase_percentage = 5

if use_AutoL2:
    Lambda_L2 = 0.1
else:
    Lambda_L2 = 0.0001

print(test_name)
print(netid)
print('num_epochs:', num_epochs)
print('batch_size:', batch_size)
print('learning_rate:', learning_rate)
print('batch_norm:', batch_norm)
print('use_AutoL2:', use_AutoL2)
print('Lambda_L2:', Lambda_L2)
print('decay_factor_L2:', decay_factor_L2)
print('increase_percentage:', increase_percentage)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')



# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
# /scratch/cl5592
train_dataset = torchvision.datasets.CIFAR10(root='/scratch/'+netid, train=True,
                                        download=False, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='/scratch/'+netid, train=False,
                                       download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, pin_memory=True, num_workers=2)

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
        nn.init.normal_(m.weight, mean=0.0, std= np.sqrt(2/fan_in))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 300, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(300, 300, 3)
        self.fc1 = nn.Linear(300 * 6 * 6, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 300, 15, 15
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 300, 6, 6
        x = x.view(-1, 300 * 6 * 6)            # -> n, 300*6*6
        x = F.relu(self.fc1(x))               # -> n, 500
        x = self.fc2(x)                       # -> n, 10
        return x

class ConvNetBN(nn.Module):
    def __init__(self):
        super(ConvNetBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 300, 3)
        self.bn1 = nn.BatchNorm2d(300)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(300, 300, 3)
        self.bn2 = nn.BatchNorm2d(300)
        self.fc1 = nn.Linear(300 * 6 * 6, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> n, 300, 15, 15
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> n, 300, 6, 6
        x = x.view(-1, 300 * 6 * 6)            # -> n, 300*6*6
        x = F.relu(self.fc1(x))               # -> n, 500
        x = self.fc2(x)                       # -> n, 10
        return x


if batch_norm == True:
    model = ConvNetBN().to(device)
else:
    model = ConvNet().to(device)


model.apply(initialize_weights)

#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        bareloss = F.cross_entropy(outputs, labels)
        L2 = L2_penalty(model.parameters(), Lambda_L2)
        loss = bareloss + L2

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        
        if (i+1) % k == 0:

            epochs.append(epoch)
            steps.append(i+1)
            #get test result
            test_result = get_test_result(model, test_loader, Lambda_L2, device)
            test_bareloss.append(test_result[0])
            test_loss.append(test_result[1])
            test_acc.append(test_result[2])

            train_bareloss.append(bareloss.item())
            train_loss.append(loss.item())
            train_acc.append(get_train_acc(outputs, labels))

            Lambda_L2s.append(Lambda_L2)

            if use_AutoL2:

                if len(train_bareloss) > 2 and i > min_step:
                    if loss_or_error_increase_by_percentage(train_bareloss[-1], train_acc[-1], min_loss, max_acc, increase_percentage) \
                        and loss_or_error_increase_by_percentage(train_bareloss[-2], train_acc[-2], min_loss, max_acc, increase_percentage):

                        Lambda_L2 = Lambda_L2 * decay_factor_L2
                        min_step = 0.1/Lambda_L2 + i   # why 0.1?

                    elif loss_or_error_increase_by_percentage(min_loss, max_acc, train_bareloss[-1], train_acc[-1], increase_percentage) \
                        and loss_or_error_increase_by_percentage(min_loss, max_acc, train_bareloss[-2], train_acc[-2], increase_percentage):

                        Lambda_L2 = Lambda_L2 / decay_factor_L2
                        min_step = 0.1/Lambda_L2 + i   # why 0.1?
                        min_loss, max_acc = train_bareloss[-2], train_acc[-2]



                elif len(train_bareloss) >= 2:
                    try:
                        min_loss, max_acc = min(train_bareloss[-2], min_loss), max(train_acc[-2], max_acc)
                    except NameError:
                        min_loss, max_acc = train_bareloss[-2], train_acc[-2]
            
            model.train()

        
        if (i+1) % 100 == 0:
            print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3,1), "GB")
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, Test Loss: {test_loss[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}')

print('Finished Training')
PATH = '/scratch/' + netid + '/'
torch.save(model.state_dict(), PATH + test_name + '.pth')


record = {
    "epochs": epochs,
    "steps": steps,
    "Lambda_L2s": Lambda_L2s,
    'train_bareloss': train_bareloss,
    'train_loss': train_loss,
    'train_acc': train_acc,
    'test_bareloss': test_bareloss,
    'test_loss': test_loss,
    'test_acc': test_acc
}

record_df = pd.DataFrame(record)
record_df['id'] = test_name
record_df['num_epochs'] = num_epochs
record_df['batch_size'] = batch_size
record_df['batch_norm'] = batch_norm
record_df['learning_rate'] = learning_rate


import os.path
fpath = PATH + test_name + 'record.csv'
header_flag = False if (os.path.exists(fpath) and (os.path.getsize(fpath) > 0)) else True

# write to csv
record_df.to_csv(fpath, mode = 'a', header = header_flag, index = False)

