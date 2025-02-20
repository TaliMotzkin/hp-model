import torch
import torch.nn.functional as F
from einops import rearrange
from skrl.models.torch import Model
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :].to(x.device)


class TransformerQNetwork(Model):
    def __init__(
        self, observation_space, action_space, device, d_model=32, nhead=4, num_layers=2
    ):
        super().__init__(observation_space, action_space, device)

        # Flattened one-hot encoded observation
        self.observation_dim = observation_space.shape[0] * 4
        self.num_actions = action_space.n

        self.fc1 = nn.Linear(4, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )
        self.fc2 = nn.Linear(d_model, self.num_actions)

        self.pos_enc = PositionalEncoding(d_model)

    def compute(self, inputs, role):
        one_hot_states = F.one_hot(inputs["states"].long(), num_classes=4)
        x = self.fc1(one_hot_states.float())
        x = self.pos_enc(x)
        x = F.relu(self.transformer(x))
        q_values = self.fc2(x)

        return q_values, None, {}

    def act(self, inputs, role):
        one_hot_states = F.one_hot(inputs["states"].long(), num_classes=4)
        x = self.fc1(one_hot_states.float())
        x = self.pos_enc(x)
        x = F.relu(self.transformer(x))
        q_values = self.fc2(x)

        return q_values[:, -1], None, {}
