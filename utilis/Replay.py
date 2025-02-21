import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store experience in buffer"""

        if state.ndim == 1:
            state = np.expand_dims(state, axis=-1)
        if next_state.ndim == 1:
            next_state = np.expand_dims(next_state, axis=-1)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample a batch of experiences"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
