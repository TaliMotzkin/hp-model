import torch.nn as nn
import torch
import torch.nn.functional as F


class LSTMDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMDQN, self).__init__()

        # self.feature_extractor = nn.Sequential(
        #     nn.Linear(input_dim, 32),  # Map actions into 32-dimensional space
        #     nn.ReLU(),
        #     nn.Linear(32, 16),  # Reduce to 16 dimensions
        #     nn.ReLU()
        # )
        self.num_layers = 2
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=6,  # in their code action_depth + hp_depth + energy_depth = 4+2+0
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.fc1 = nn.Linear(self.hidden_dim, 3)
        # self.hidden = None

    def reset_hidden(self, batch_size):
        """Manually reset hidden state before a new episode starts."""
        device = next(self.parameters()).device
        # h_0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        # c_0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        # self.hidden = (h_0.detach(), c_0.detach())

    def forward(self, x, x_seq):
        x = x.to(next(self.parameters()).device)
        x_seq = x_seq.to(x.device)

        one_hot_actions = F.one_hot(x.long(), num_classes=4)
        one_hot_sequence = F.one_hot(x_seq.long(), num_classes=2)
        one_hot_input = torch.cat([one_hot_actions, one_hot_sequence], dim=-1)

        batch_size = x.shape[0]
        hidden_states = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        lstm_out, _ = self.lstm(one_hot_input.float(), (hidden_states, hidden_states.clone()))

        out = self.fc1(lstm_out[:, -1, :])

        return out
