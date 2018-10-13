# https://github.com/spro/char-rnn.pytorch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharRNN2(nn.Module):
    def __init__(self, hidden_size, model, char_dict, n_layers=2):
        assert model in ["gru", "lstm"]

        super(CharRNN2, self).__init__()
        self.model = model.lower()
        self.char_dict = char_dict
        self.input_size = len(char_dict)
        self.n_fac = math.floor(self.input_size * .7)
        self.hidden_size = hidden_size
        self.output_size = len(char_dict)
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.n_fac)
        if self.model == "gru":
            self.rnn = nn.GRU(self.n_fac, self.hidden_size, self.n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(self.n_fac, self.hidden_size, self.n_layers)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = F.log_softmax(self.decoder(output.view(batch_size, -1)), dim=-1)
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size))
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

    # Turning a string into a tensor
    def char2tensor(self, s):
        i_in = [self.char_dict.find(s[i]) for i in range(0, len(s))]
        # replacing -1 with 0
        i_in = [i if i >= 0 else 0 for i in i_in]

        return torch.LongTensor(i_in)
