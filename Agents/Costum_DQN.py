import torch.nn.functional as F
import numpy as np
from utilis.Replay import ReplayBuffer, PrioritizedReplayBuffer
import torch.optim as optim
from models.LASTM_costum import LSTMDQN
import torch


class DQNAgent:
    def __init__(self, env, hidden_dim=256, gamma=0.98, lr=1e-3, batch_size=32, buffer_capacity=25000, seed=0):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.epsilon = 0.995
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env.action_space.seed(seed)

        # input_dim = env.observation_space.shape[0]  #length
        input_dim = 1  # length
        output_dim = env.action_space.n  # of actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = LSTMDQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_network = LSTMDQN(input_dim, hidden_dim, output_dim).to(self.device)
        # self.target_network.load_state_dict(self.q_network.state_dict())  # init target network
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        print("env.observation_space", env.observation_space[0])
        self.buffer = ReplayBuffer(buffer_capacity, env.observation_space[0].shape[0], self.device)

    def select_action_1(self, state_action, state_seq):
        """Select an action using epsilon-greedy"""
        np.random.seed(self.seed)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        if isinstance(state_action, torch.Tensor):
            state_tensor = state_action.clone().detach().unsqueeze(0).to(self.device)
            state_tensor_seq = state_seq.clone().detach().unsqueeze(0).to(self.device)
        else:
            state_tensor = torch.tensor(state_action, dtype=torch.float32).unsqueeze(0).to(self.device)
            state_tensor_seq = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # print("state_tensor", state_tensor.shape)
            q_values = self.q_network(state_tensor, state_tensor_seq)

        return torch.argmax(q_values).item()

    def train_step(self):
        """Perform a training step using experience replay"""
        if len(self.buffer) < self.batch_size:
            # print(len(self.buffer))
            return

        batch = self.buffer.sample(self.batch_size)
        # print(f"Batch type: {type(batch)}")  # Should be a tuple
        # print(f"Batch length: {len(batch)}")  # Should be 5
        # print(f"Batch content example: {batch[2:]}")  # Print first few sample
        states_actions, state_seq, actions, rewards, next_states_actions, next_states_seq, dones = batch

        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        # actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1).to(self.device)
        # rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(self.device)
        # next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        # dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(self.device)

        # print("rewards", rewards.shape)  # Size([64, 1])
        # print("dones", dones.shape)  # Size([64, 1])
        # print("actions", actions.shape)  # Size([64, 1])
        # print("states", states.shape)  # torch.Size([64, 46, 1])
        # print("next_states", next_states.shape)  # torch.Size([64, 46, 1])

        # Compute current Q-values
        # if states.dim() == 2:  # If shape is (batch_size, seq_length)
        #     states = states.unsqueeze(-1)  # Convert to (batch_size, seq_length, 1)
        # if next_states.dim() == 2:
        #     next_states = next_states.unsqueeze(-1)

        q_values = self.q_network(states_actions, state_seq)
        q_values = q_values.gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states_actions, next_states_seq).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values  # Size([64, 1]

        # print("next_q_values", next_q_values.shape)
        bellman_error = target_q_values - q_values
        # weighted_bellman_error = weights * bellman_error
        clipped_bellman_error = bellman_error.clamp(-1, 1)
        d_error = clipped_bellman_error * -1.0

        # print(f"Q-values shape: {q_values.shape}")  # Expected: [64, 1]
        # print(f"bellman_error: {bellman_error.shape}")  # Unexpected: [64, 64, 1]
        # print("target_q_values", target_q_values.shape)

        self.optimizer.zero_grad()
        q_values.backward(d_error.data)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        # self.buffer.update_priorities(indices, bellman_error.abs())
        # # Compute loss
        # loss = F.mse_loss(q_values, target_q_values)
        #
        # # Optimize the model
        # self.optimizer.zero_grad()
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        #
        # loss.backward()
        # self.optimizer.step()

    def update_target_network(self):
        """Copy parameters from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Detach hidden state to prevent backward graph issues

