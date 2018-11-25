import numpy as np
import random
from collections import deque
import torch

class ExpertBuffer:

    def __init__(self, size):
        self.buffer = deque(maxlen=int(size))
        self.maxSize = size
        self.len = 0
    
    def __len__(self):
        return self.len
    
    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        
        return torch.from_numpy(np.float32(batch))

    def len(self):
        return self.len

    def add(self, state_action):
        """
        adds a particular transaction in the memory buffer
        :param state_action(np.ndarray): current state_action
        :return:
        """
        transition = state_action
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)
