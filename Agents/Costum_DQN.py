import torch.nn.functional as F
import numpy as np
from utilis.Replay import ReplayBuffer
import torch.optim as optim
from models.LASTM_costum import LSTMDQN
import torch

class DQNAgent:
    def __init__(self, env, hidden_dim=512, gamma=0.98, lr=0.0005, batch_size=32, buffer_capacity=5000, seed=0):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon_max = 1
        self.decay_rate = 5
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env.action_space.seed(seed)

        # input_dim = env.observation_space.shape[0]  #length
        input_dim = 1  #length
        output_dim = env.action_space.n  # of actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = LSTMDQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_network = LSTMDQN(input_dim, hidden_dim, output_dim).to(self.device)
        # self.target_network.load_state_dict(self.q_network.state_dict())  # init target network
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, env.observation_space.shape[0],self.device)

    def select_action_1(self, state):
        """Select an action using epsilon-greedy"""
        np.random.seed(self.seed)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        if isinstance(state, torch.Tensor):
            state_tensor = state.clone().detach().unsqueeze(0).unsqueeze(-1).to(self.device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            # print("state_tensor", state_tensor.shape)
            q_values = self.q_network(state_tensor)

        return torch.argmax(q_values).item()

    def train_step(self):
        """Perform a training step using experience replay"""
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        # print(f"Batch type: {type(batch)}")  # Should be a tuple
        # print(f"Batch length: {len(batch)}")  # Should be 5
        # print(f"Batch content example: {batch[2:]}")  # Print first few sample
        states, actions, rewards, next_states, dones = batch
        # print("rewards", rewards)
        
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        dones = dones.unsqueeze(1).to(self.device)


        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        bellman_error = target_q_values - q_values
        
        clipped_bellman_error = bellman_error.clamp(-1, 1)
        d_error = clipped_bellman_error * -1.0

        loss_value = F.smooth_l1_loss(q_values, target_q_values)
        
    
        self.optimizer.zero_grad()
        q_values.backward(d_error.data)

        self.optimizer.step()
    
        # loss = F.mse_loss(q_values, target_q_values)

        # # Optimize the model
        # self.optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0) 
        # self.optimizer.step()
        loss_value = F.smooth_l1_loss(q_values, target_q_values)
        return loss_value.item()

    def update_target_network(self):
        """Copy parameters from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

