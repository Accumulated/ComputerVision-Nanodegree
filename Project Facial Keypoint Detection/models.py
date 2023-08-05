## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout_prob=0.5):
        super(ConvBlock, self).__init__()
        
        # Convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Activation function
        self.activation = nn.ReLU(inplace=True)

        self.bn = nn.BatchNorm2d(out_channels)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)
        x = self.pool(x)
        return x


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional block
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=0, dropout_prob=0.1)
        self.conv_block2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, dropout_prob=0.2)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0, dropout_prob=0.3)
        self.conv_block4 = ConvBlock(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, dropout_prob=0.4)
        
        # Fully connected layers, activations and dropouts
        self.fc1 = nn.Linear(256*13*13, 1000)

        self.fc2 = nn.Linear(1000, 1000)

        self.fc3 = nn.Linear(1000, 136)

        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        
        return x

