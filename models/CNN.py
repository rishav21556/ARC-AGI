import torch
import torch.nn as nn
from torch import flatten
from torch.utils.data import DataLoader
from torchvision import transforms

# Define a transform to convert the images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalization parameters for MNIST
])

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


def getConvolutionLayer(inChannel, outChannel, kernel_size, padding):
    convolution = nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=kernel_size, padding=padding)
    return convolution

def getPoolingLayer(kernel_size , stride):
    pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    return pooling


class Cnn(nn.Module):
    def __init__(self,inChannel = 1, convolutions = [], poolings = [], input_size = (30,30)):
        super(Cnn,self).__init__()
        self.conv = nn.ModuleList(convolutions)
        self.pool = nn.ModuleList(poolings)

        self.flattened_size = self.calculate_flattened_size(input_size=input_size, inChannel=inChannel)

        self.fc1 = nn.Linear(self.flattened_size,128)
        self.fc2 = nn.Linear(128, 64)
    
    def calculate_flattened_size(self, input_size, inChannel):
        # Pass a dummy input through the conv and pool layers to determine the size
        dummy_input = torch.zeros(1, inChannel, *input_size)  # Batch size of 1, input shape as specified
        dummy_output = self.pool1(torch.sigmoid(self.conv1(dummy_input)))  # After Conv1 + Pool1
        dummy_output = self.pool2(torch.sigmoid(self.conv2(dummy_output)))  # After Conv2 + Pool2
        
        # Calculate the flattened size of the output
        flattened_size = dummy_output.numel()  # This gives total elements in the feature map
        
        return flattened_size
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = self.pool2(x)
        
        x = flatten(x,1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.softmax(x, dim = 1)
        
        return x
        
        
        