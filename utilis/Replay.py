import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, state_dim, device, seed=0):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = torch.device(device)
        self.seed = seed
        random.seed(seed)

        self.states_actions = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.states_seq = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)

        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states_actions = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.next_states_seq = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)

        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def push(self, state_actions, state_seq, action, reward, next_state_actions, next_state_seq, done):
        """Store experience as PyTorch tensors on GPU"""
        # if isinstance(state, torch.Tensor):
        #     state = state.clone().detach().to(self.device)
        # else:
        #     state = torch.tensor(state, dtype=torch.float32, device=self.device)
        #
        # if isinstance(next_state, torch.Tensor):
        #     next_state = next_state.clone().detach().to(self.device)
        # else:
        #     next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

        # if state.dim() == 1:
        #     state = state.unsqueeze(-1)

        # if next_state.ndim == 1:
        #     next_state = next_state.unsqueeze(-1)

        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        # if len(self.buffer) < self.capacity:
        #     self.buffer.append(None)
        # self.buffer[self.position] = (state, action, reward, next_state, done)
        # self.position = (self.position + 1) % self.capacity  # Circular buffer

        index = self.position % self.capacity

        self.states_actions[index] = state_actions.clone().detach().to(self.device)
        self.states_seq[index] = state_seq.clone().detach().to(self.device)

        self.actions[index] = torch.tensor(action, device=self.device)
        self.rewards[index] = torch.tensor(reward, device=self.device)

        self.next_states_actions[index] = next_state_actions.clone().detach().to(self.device)
        self.next_states_seq[index] = next_state_seq.clone().detach().to(self.device)

        self.dones[index] = torch.tensor(done, device=self.device)

        self.position += 1

    def sample(self, batch_size):
        """Randomly sample a batch and return tensors on GPU"""
        # random.seed(self.seed)
        # torch.manual_seed(self.seed)
        # torch.cuda.manual_seed(self.seed)
        # torch.cuda.manual_seed_all(self.seed)

        # batch = random.sample(self.buffer, batch_size)
        #
        # # Unzip data correctly
        # states, actions, rewards, next_states, dones = zip(*batch)
        #
        # return (
        #     torch.stack(states),  # Keep batch as tensors
        #     torch.tensor(actions, dtype=torch.long, device=self.device),
        #     torch.stack(rewards),
        #     torch.stack(next_states),
        #     torch.stack(dones)
        # )
        max_mem = min(self.position, self.capacity)
        # print("max_mem", max_mem)
        batch_indices = torch.randint(0, max_mem, (batch_size,), device=self.device)
        # if max_mem >490:
        #     print("batch_indices", batch_indices)

        return (
            self.states_actions[batch_indices],
            self.states_seq[batch_indices],
            self.actions[batch_indices],
            self.rewards[batch_indices],
            self.next_states_actions[batch_indices],
            self.next_states_seq[batch_indices],
            self.dones[batch_indices]
        )

    def __len__(self):
        return min(self.position, self.capacity)

        # return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_dim, device, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.states = torch.zeros((capacity, state_dim, 1), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.priorities = torch.ones((capacity,), dtype=torch.float32, device=device)

    def push(self, state, action, reward, next_state, done, td_error=1.0):
        """Efficiently store experience in a tensor-based prioritized buffer"""
        index = self.position % self.capacity

        if state.dim() == 1:
            state = state.unsqueeze(-1)

        if next_state.ndim == 1:
            next_state = next_state.unsqueeze(-1)

        self.states[index] = state.clone().detach().to(self.device)
        self.actions[index] = torch.tensor(action, device=self.device)
        self.rewards[index] = torch.tensor(reward, device=self.device)
        self.next_states[index] = next_state.clone().detach().to(self.device)
        self.dones[index] = torch.tensor(done, device=self.device)

        # Assign priority (new experiences get max priority)
        self.priorities[index] = (abs(td_error) + 1e-5) ** self.alpha

        self.position += 1

    def sample(self, batch_size):
        """Sample experiences based on priority (GPU-optimized)"""
        total_priority = self.priorities[:min(self.position, self.capacity)].sum()

        probabilities = self.priorities[:min(self.position, self.capacity)] / total_priority

        # based on probability
        indices = torch.multinomial(probabilities, batch_size, replacement=False)

        # importance-sampling weights
        weights = (len(self.priorities) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights.unsqueeze(1),
            indices
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities of sampled experiences"""
        self.priorities[indices] = (abs(td_errors) + 1e-5) ** self.alpha

    def __len__(self):
        return min(self.position, self.capacity)
