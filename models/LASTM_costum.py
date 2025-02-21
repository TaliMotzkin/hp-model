import torch.nn as nn

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

    def forward(self, x):
        #x = x.unsqueeze(-1).float()
        x = self.feature_extractor(x)
        lstm_out, _ = self.lstm(x)
        q_values = self.fc(lstm_out[:, -1, :])
        return q_values
