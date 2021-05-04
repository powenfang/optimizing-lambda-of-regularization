import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_name = "simple_cnn_test"
netid = "cl5592"
# Hyper-parameters 
num_epochs = 5 #6
batch_size = 128 #128
learning_rate = 0.001

# Global-variables for storing CSV
record = {
    "id": test_name,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
}

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
# /scratch/cl5592
train_dataset = torchvision.datasets.CIFAR10(root='/scratch/cl5592', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='/scratch/cl5592', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

# example structure of CNN
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # -> n, 3, 32, 32
#         x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
#         x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
#         x = x.view(-1, 16 * 5 * 5)            # -> n, 400
#         x = F.relu(self.fc1(x))               # -> n, 120
#         x = F.relu(self.fc2(x))               # -> n, 84
#         x = self.fc3(x)                       # -> n, 10
#         return x


model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = '/scratch/' + netid + '/'
torch.save(model.state_dict(), PATH + test_name + '.pth')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        # for i in range(batch_size):
        for i in range(len(labels))
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
    record["Accuracy_all"] = acc;

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
        record["Accuracy_"+classes[i]] = acc

# write to csv
record = pd.DataFrame(record, index=[0])
record.to_csv(PATH + 'record.csv', mode='a', header=True)

