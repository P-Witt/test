import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

"""Simple Mnist Classifier"""

"""Hyper Parameters"""
batch_size = 128
in_dim = 28*28
img_size = 28
num_classes = 10
"""Data Loading"""
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])
data = datasets.MNIST(root="dataset/", transform=transforms, download=False)
length = len(data)
trainset, val_set = torch.utils.data.random_split(data, [int(len(data)*2/3),int(len(data)*1/3)])
train_loader = DataLoader(trainset, batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle= True)
"""Model"""

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,2,2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(2,2,4),
            nn.MaxPool2d(2),
            nn.ReLU(),)
        self.linear =nn.Sequential(
            nn.Linear(2*5*5, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,num_classes),
            nn.ReLU()
        )
    def forward(self,input):
        input  =input.view(-1,1,img_size,img_size)
        y = self.conv(input)
        y =y.view(-1, 2*5*5)
        y = self.linear(y)
        return y

classifier = Classifier(in_dim, num_classes).to(device="cuda")

"""Training"""
optim = optim.Adam(classifier.parameters(),0.0001)
criterion = nn.MSELoss()
epochs = 10
writer = SummaryWriter(f"logs_classifier")
index = 0
for epoch in range(epochs):
    for batch_idx, (input, labels_idx) in enumerate(train_loader):
        optim.zero_grad()
        labels = torch.zeros(labels_idx.shape[0], num_classes)
        for i in range(labels_idx.shape[0]):
            labels[i,labels_idx[i]] =1
        labels = labels.to(device="cuda")
        output = classifier(input.to(device="cuda"))
        loss = criterion(output,labels)
        loss.backward()
        optim.step()
        _,prediction = torch.max(output, dim = 1)
        if batch_idx%100 == 0:
            (val_input, val_labels_idx) = next(iter(val_loader))
            val_output = classifier(val_input.to("cuda"))
            _, val_prediction = torch.max(val_output, dim= 1)
            val_accuracy = torch.sum(val_labels_idx.to("cuda")==val_prediction)/val_labels_idx.shape[0]*100
            accuracy = torch.sum(labels_idx.to("cuda") == prediction).item()/labels_idx.shape[0]*100
            print(f"Epoch[{epoch} {batch_idx}/{epochs}]: Loss: {loss}, Accuracy: {accuracy}, Val_Accuracy: {val_accuracy}")
            index = index +1
            writer.add_scalar("loss",loss, index)
            writer.add_scalar("Accuracy", accuracy, index)
            writer.add_scalar("Val_Accuracy",val_accuracy, index)
