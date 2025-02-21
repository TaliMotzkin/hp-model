import torch.nn.functional as F
import numpy as np
from utilis.Replay import ReplayBuffer
import torch.optim as optim
from models.LASTM_costum import LSTMDQN
import torch

class DQNAgent:
    def __init__(self, env, hidden_dim=128, gamma=0.99, lr=1e-3, batch_size=64, buffer_capacity=500, seed=0):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env.action_space.seed(seed)

        # input_dim = env.observation_space.shape[0]  #length
        input_dim = 1  #length
        output_dim = env.action_space.n  # of actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = LSTMDQN(input_dim, hidden_dim, output_dim)
        self.target_network = LSTMDQN(input_dim, hidden_dim, output_dim)
        # self.target_network.load_state_dict(self.q_network.state_dict())  # init target network
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    def select_action_1(self, state):
        """Select an action using epsilon-greedy"""
        np.random.seed(self.seed)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)   # Add batch dimension
        with torch.no_grad():
            # print("state_tensor", state_tensor.shape)
            q_values = self.q_network(state_tensor)

        return torch.argmax(q_values).item()

    def train_step(self):
        """Perform a training step using experience replay"""
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute current Q-values
        # if states.dim() == 2:  # If shape is (batch_size, seq_length)
        #     states = states.unsqueeze(-1)  # Convert to (batch_size, seq_length, 1)
        # if next_states.dim() == 2:
        #     next_states = next_states.unsqueeze(-1)
        # print("states", states)
        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Copy parameters from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Detach hidden state to prevent backward graph issues

