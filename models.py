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



class AutoMaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(AutoMaskLSTM, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        #counting non -1 in first feature column
        seq_lengths = (x[:, :, 0] != -1).sum(dim=1).cpu()

        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        x = x[perm_idx]

        x_packed = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=True)

        lstm_out, _ = self.lstm(x_packed)

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Restore original order
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]

        # Use the last valid hidden state
        batch_size = x.shape[0]
        last_valid_idx = (seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, lstm_out.size(2))
        last_hidden_states = lstm_out.gather(1, last_valid_idx).squeeze(1)

        return self.fc(last_hidden_states)

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