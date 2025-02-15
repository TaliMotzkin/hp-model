from typing import Mapping, Union, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, DeterministicMixin


# define the model
class RNN_class(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=1, hidden_size=256, sequence_length=20):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations, #in their code action_depth + hp_depth + energy_depth = 4+2+0
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)

        self.fc1 = nn.Linear(self.hidden_size , self.num_actions)


    def compute(self, inputs, role):
        states = inputs["states"]


        # critic models are only used during training
        rnn_input = states.view(-1, self.sequence_length,
                                states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length

        batch_size = rnn_input.size(0) // self.sequence_length
        hidden_states = torch.zeros(self.num_layers * 1, batch_size, self.hidden_size)  # (D * num_layers, N, L, Hout)

        rnn_output, h_0 = self.lstm(rnn_input, (hidden_states, hidden_states.clone()))

        # flatten the RNN output
        rnn_output =  rnn_output.view(-1, rnn_output.shape[-1])   # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.fc1(rnn_output), {}


