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
        self.drop = nn.Dropout(p=0.4)
        
        # Convolutional Blocks --> Conv --> Activation --> Maxpool --> Dropout
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        
        # Dense Block
        self.dense1 = nn.Linear(43264, 2048)
        self.dense2 = nn.Linear(2048, 1000)
        self.dense3 = nn.Linear(1000, 136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Convolutional Block
        x = self.drop(self.pool2d(F.relu(self.conv1(x))))
        x = self.drop(self.pool2d(F.relu(self.conv2(x))))
        x = self.drop(self.pool2d(F.relu(self.conv3(x))))
        x = self.drop(self.pool2d(F.relu(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense Block
        x = self.drop(F.relu(self.dense1(x)))
        x = self.drop(F.relu(self.dense2(x)))
        x = self.dense3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
