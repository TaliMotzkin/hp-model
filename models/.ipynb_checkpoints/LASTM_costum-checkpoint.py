import torch.nn as nn
import torch


class LSTMDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMDQN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),  # Map actions into 32-dimensional space
            nn.ReLU(),
            nn.Linear(32, 16),  # Reduce to 16 dimensions
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=16, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.hidden = None

    def reset_hidden(self, batch_size):
        """Manually reset hidden state before a new episode starts."""
        device = next(self.parameters()).device
        h_0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        self.hidden = (h_0.detach(), c_0.detach())

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = self.feature_extractor(x)

        if self.hidden is None or x.shape[0] != self.hidden[0].shape[1]:
            self.reset_hidden(batch_size=x.shape[0])
        # print("hidden ", self.hidden)
        lstm_out, hidden = self.lstm(x, self.hidden)
        self.hidden = (hidden[0].detach(), hidden[1].detach())
        q_values = self.fc(lstm_out[:, -1, :])
        return q_values
