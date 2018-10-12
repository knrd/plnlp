# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model, n_layers=2):
        assert model in ["gru", "lstm"]

        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size))
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)
