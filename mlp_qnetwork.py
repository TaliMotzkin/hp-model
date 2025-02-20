import torch.nn.functional as F
from skrl.models.torch import Model
from torch import nn
from einops import rearrange


class MLPQNetwork(Model):
    def __init__(self, observation_space, action_space, device, d_model=128):
        super().__init__(observation_space, action_space, device)

        # Flattened one-hot encoded observation
        self.observation_dim = observation_space.shape[0] * 4
        self.num_actions = action_space.n

        self.fc1 = nn.Linear(self.observation_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, self.num_actions)

    def compute(self, inputs, role):
        one_hot_states = F.one_hot(inputs["states"].long(), num_classes=4)
        one_hot_states = rearrange(one_hot_states, "b s d -> b (s d)")
        x = F.relu(self.fc1(one_hot_states.float()))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values, {}

    def act(self, inputs, role):
        one_hot_states = F.one_hot(inputs["states"].long(), num_classes=4)
        one_hot_states = rearrange(one_hot_states, "b s d -> b (s d)")
        x = F.relu(self.fc1(one_hot_states.float()))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values, None, {}
