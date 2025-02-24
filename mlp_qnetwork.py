import torch.nn.functional as F
from einops import rearrange
from skrl.models.torch import Model
from torch import nn
import torch


class MLPQNetwork(Model):
    def __init__(self, observation_space, action_space, device, d_model=128):
        super().__init__(observation_space, action_space, device)

        # Flattened one-hot encoded observation
        self.sequence_length = observation_space[0].shape[0]
        self.observation_dim = observation_space[0].shape[0] * 4 + observation_space[1].shape[0] * 2
        self.num_actions = action_space.n

        self.fc1 = nn.Linear(self.observation_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, self.num_actions)

    def compute(self, inputs, role):
        observation_actions, observation_sequence = inputs["states"][..., :self.sequence_length], inputs["states"][..., self.sequence_length:]

        one_hot_actions = F.one_hot(observation_actions.long(), num_classes=4)
        one_hot_actions = rearrange(one_hot_actions, "b s d -> b (s d)")
        one_hot_sequence = F.one_hot(observation_sequence.long(), num_classes=2)
        one_hot_sequence = rearrange(one_hot_sequence, "b s d -> b (s d)")
        one_hot_input = torch.cat([one_hot_actions, one_hot_sequence], dim=-1)

        x = F.relu(self.fc1(one_hot_input.float()))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values, {}

    def act(self, inputs, role):
        observation_actions, observation_sequence = inputs["states"][..., :self.sequence_length], inputs["states"][..., self.sequence_length:]

        one_hot_actions = F.one_hot(observation_actions.long(), num_classes=4)
        one_hot_actions = rearrange(one_hot_actions, "b s d -> b (s d)")
        one_hot_sequence = F.one_hot(observation_sequence.long(), num_classes=2)
        one_hot_sequence = rearrange(one_hot_sequence, "b s d -> b (s d)")
        one_hot_input = torch.cat([one_hot_actions, one_hot_sequence], dim=-1)

        x = F.relu(self.fc1(one_hot_input.float()))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values, None, {}
