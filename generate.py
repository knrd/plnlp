#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import argparse

from model import CharRNN2


def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, cuda=False):
    decoder.eval()
    with torch.no_grad():
        hidden = decoder.init_hidden(1)
        prime_input = decoder.char2tensor(prime_str).unsqueeze(0)

        if cuda:
            if decoder.model == "lstm":
                hidden = (hidden[0].cuda(), hidden[1].cuda())
            else:
                hidden = hidden.cuda()
            prime_input = prime_input.cuda()
        predicted = prime_str

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str) - 1):
            _, hidden = decoder(prime_input[:, p], hidden)

        inp = prime_input[:, -1]

        for p in range(predict_len):
            output, hidden = decoder(inp, hidden)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_char = decoder.char_dict[top_i]
            predicted += predicted_char
            inp = decoder.char2tensor(predicted_char).unsqueeze(0)
            if cuda:
                inp = inp.cuda()

    return predicted


# Run as standalone script
if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    # argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    state = torch.load(args.filename)
    assert state['version'] == (0, 1)
    decoder = CharRNN2(state['model_hidden_size'], state['model_type'], state['model_char_dict'], state['model_n_layers'])
    decoder.load_state_dict(state['model_state'])
    del args.filename
    print(generate(decoder, **vars(args)))
