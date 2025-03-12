import random

import torch
from mamba_ssm import Mamba, Mamba2
from torch import nn


class RNN_LSTM_onlyLastHidden(nn.Module):
    """
    LSTM version that just uses the information from the last hidden state
    since the last hidden state has information from all previous states
    basis for BiDirectional LSTM
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(RNN_LSTM_onlyLastHidden, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        # change basic RNN to LSTM
        # num_layers Default: 1
        # bias Default: True
        # batch_first Default: False
        # dropout Default: 0
        # bidirectional Default: False
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # remove the sequence_length
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Get data to cuda if possible
        x = x.to(self.device)
        # print("input x.size() = ", x.size(), x)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # LSTM needs a separate cell state (LSTM needs both hidden and cell state)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        # need to give LSTM both hidden and cell state (h0, c0)
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # print("lstm out", out)
        # Decode the hidden state of the last time step
        # no need to reshape the out or concat
        # out is going to take all mini-batches at the same time + last layer + all features
        out = self.fc(out[:, -1, :])
        # print("forward out = ", out)
        return out

    def sample_action(self, obs, epsilon):
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            explore_action = random.randint(0, 2)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()


class MambaModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(MambaModel, self).__init__()
        self.hidden_size = hidden_size
        self.expander = nn.Linear(input_size, self.hidden_size)
        self.mamba_layers = nn.Sequential(
            *[
                Mamba(d_model=hidden_size, d_state=64, d_conv=2, expand=2)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = self.expander(x)
        x = self.mamba_layers(x)
        out = self.fc(x[:, -1, :])

        return out

    def sample_action(self, obs, epsilon):
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            explore_action = random.randint(0, 2)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()
