import torch as torch 
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    '''
    The MLP class maps a feature vector to an output vector of scores for each action.
    '''
    def __init__(self, input_size:tuple[int], action_size:int, hidden_size:int=256):
        super(MLP, self).__init__()
        """
        input: tuple[int]
            The input size of the image, of shape (channels, height, width)
        action_size: int
            The number of possible actions
        hidden_size: int
            The number of neurons in the hidden layer
        """
        C, H, W = input_size
        self.linear1 = nn.Linear(C*H*W, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, action_size)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = self.linear1(x)
        out = self.relu(out)
        out = self.output(out)
        return out
    
class CNN(nn.Module):
    """
    A class that defines a neural network with the following architecture:
    - 1 convolutional layer with 32 8x8 kernels with a stride of 4x4 w/ ReLU activation
    - 1 convolutional layer with 64 4x4 kernels with a stride of 2x2 w/ ReLU activation
    - 1 convolutional layer with 64 3x3 kernels with a stride of 1x1 w/ ReLU activation
    - 1 fully connected layer with 512 neurons and ReLU activation. 
    Based on 2015 paper 'Human-level control through deep reinforcement learning' by Mnih et al
    """
    def __init__(self, input_size:tuple[int], action_size:int,**kwargs):
        super(CNN, self).__init__()
        """
        input: tuple[int]
            The input size of the image, of shape (channels, height, width)
        action_size: int
            The number of possible actions
        **kwargs: dict
            additional kwargs to pass for stuff like dropout, etc if you would want to implement it
        """
        C,_,_ = input_size
        self.conv = nn.Sequential(*[
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        ])
        self.flatten = nn.Flatten(start_dim=1)
        self.MLP = MLP(input_size=(64,7,7),action_size=action_size,hidden_size=512)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = self.conv(x)
        out = self.flatten(out)
        out = self.MLP(out)
        return out