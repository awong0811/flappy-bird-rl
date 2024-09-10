import torch 
import torch.optim as optim
import torch.nn.functional as F
import torch.nn
import gymnasium as gym
from replay_buffer import ReplayBufferDQN
import random
import numpy as np
import os
import time
from utils import exponential_decay
import typing

class DQN:
    def __init__(self, env:typing.Union[gym.Env,gym.Wrapper],
                 )