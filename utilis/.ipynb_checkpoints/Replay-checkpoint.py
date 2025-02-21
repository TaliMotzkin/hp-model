import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_dim, device,seed=0):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = torch.device(device)
        self.seed = seed
        random.seed(seed)

        self.states = torch.zeros((capacity, state_dim,1), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim,1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def push(self, state, action, reward, next_state, done):
        """Store experience as PyTorch tensors on GPU"""
        if isinstance(state, torch.Tensor):
            state = state.clone().detach().to(self.device)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.clone().detach().to(self.device)
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

        if state.dim() == 1:
            state = state.unsqueeze(-1)

        if next_state.ndim == 1:
            next_state = next_state.unsqueeze(-1)


        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity  # Circular buffer
        # index = self.position % self.capacity

        # self.states[index] = state.clone().detach().to(self.device)
        # self.actions[index] = torch.tensor(action , device=self.device)
        # self.rewards[index] = torch.tensor(reward , device=self.device)
        # self.next_states[index] = next_state.clone().detach().to(self.device)
        # self.dones[index] = torch.tensor(done , device=self.device)

        # self.position += 1

    def sample(self, batch_size):
        """Randomly sample a batch and return tensors on GPU"""
        
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip data correctly
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states),  # Keep batch as tensors
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones)
        )
        # max_mem = min(self.position, self.capacity)
        # batch_indices = torch.randint(0, max_mem, (batch_size,), device=self.device)
        # if max_mem >400:
        #     print("batch_indices", batch_indices)

        # return (
        #     self.states[batch_indices],
        #     self.actions[batch_indices],
        #     self.rewards[batch_indices],
        #     self.next_states[batch_indices],
        #     self.dones[batch_indices]
        # )

    def __len__(self):
        return len(self.buffer)
        # return min(self.position, self.capacity)