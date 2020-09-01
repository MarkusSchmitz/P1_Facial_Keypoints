## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # Maxpool and Dropout
        self.pool2d = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.4)
        self.fc1_bn = nn.BatchNorm1d(2048)
        
        # Convolutional Blocks --> Conv --> Activation --> Maxpool --> Dropout
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        
        # Dense Block
        self.dense1 = nn.Linear(18432, 2048)
        #self.dense2 = nn.Linear(2048, 1024)
        #self.dense3 = nn.Linear(2048, 1000)
        self.dense4 = nn.Linear(2048, 136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Convolutional Block
        
        x = self.pool2d(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool2d(F.relu(self.conv2(x)))
        x = self.pool2d(F.relu(self.conv3(x)))
        x = self.pool2d(F.relu(self.conv4(x)))
        x = self.pool2d(F.relu(self.conv5(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense Block
        x = self.drop5(self.fc1_bn(F.relu(self.dense1(x))))
        #x = self.drop(F.relu(self.dense2(x)))
        #x = self.drop(F.relu(self.dense3(x)))
        x = self.dense4(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
