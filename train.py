#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import argparse
import os
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from helpers import char_tensor, read_file, start_time, time_since, n_characters
from model import CharRNN
from generate import generate
from tblogger import TBLogger


class Trainer(object):
    def __init__(self, filename, model="gru", hidden_size=100, n_layers=2, cuda=True, chunk_len=200, batch_size=100, tensorboard=True, verbose=1):
        assert model in ["gru", "lstm"]

        self.verbose = verbose
        self.cuda = cuda
        self.filename = filename
        self.file, self.file_len = read_file(self.filename)
        self.start = start_time()
        self.all_losses = []
        self.loss_avg = 0
        self.chunk_len = chunk_len
        self.batch_size = batch_size
        self.tensorboard = tensorboard
        self.model = model
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.logger = None
        if self.tensorboard:
            self.logger = TBLogger('./logs')

        # Initialize models
        self.decoder = CharRNN(
            n_characters,
            self.hidden_size,
            n_characters,
            model=model,
            n_layers=self.n_layers,
        )
        self.criterion = nn.CrossEntropyLoss()

        if self.cuda:
            self.decoder.cuda()

    def _random_training_set(self):
        inp = torch.LongTensor(self.batch_size, self.chunk_len)
        target = torch.LongTensor(self.batch_size, self.chunk_len)
        for bi in range(self.batch_size):
            start_index = random.randint(0, self.file_len - self.chunk_len)
            end_index = start_index + self.chunk_len + 1
            chunk = self.file[start_index:end_index]
            inp[bi] = char_tensor(chunk[:-1])
            target[bi] = char_tensor(chunk[1:])
        inp = Variable(inp)
        target = Variable(target)
        if self.cuda:
            inp = inp.cuda()
            target = target.cuda()
        return inp, target

    def train_step(self, inp, target, decoder_optimizer):
        hidden = self.decoder.init_hidden(self.batch_size)
        if self.cuda:
            if self.model == "lstm":
                hidden = (hidden[0].cuda(), hidden[1].cuda())
            else:
                hidden = hidden.cuda()

        self.decoder.zero_grad()
        loss = 0

        for c in range(self.chunk_len):
            output, hidden = self.decoder(inp[:, c], hidden)
            loss += self.criterion(output.view(self.batch_size, -1), target[:, c])

        loss.backward()
        decoder_optimizer.step()

        _, argmax = torch.max(output, 1)
        accuracy = (target[:, c] == argmax.squeeze()).float().mean()

        return loss.data.item() / self.chunk_len, accuracy.data.item()

    def train(self, learning_rate, n_epochs=200, print_every=100):
        if self.cuda and self.verbose:
            print("Using CUDA")
        label = "m_%s-lr_%s-hs_%d-nl-%d" % (self.model, learning_rate, self.hidden_size, self.n_layers)

        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate)
        try:
            print("Training %s for %d epochs..." % (label, n_epochs))
            for epoch in tqdm(range(1, n_epochs + 1)):
                inp, target = self._random_training_set()
                loss, accuracy = self.train_step(inp, target, decoder_optimizer)
                self.loss_avg += loss

                if self.logger:
                    # 1. Log scalar values (scalar summary)
                    info = {'loss': loss, 'accuracy': accuracy, 'learning_rate': learning_rate}

                    for tag, value in info.items():
                        self.logger.scalar_summary(label, tag, value, epoch)

                if self.verbose and epoch % print_every == 0:
                    print('[%s (%d %d%%) %.4f]' % (time_since(self.start), epoch, epoch / n_epochs * 100, loss))
                    if self.verbose > 1:
                        print(generate(self.decoder, 'Wh', 100, cuda=self.cuda), '\n')

            print("Saving...")
            self.save(label)

        except KeyboardInterrupt:
            print("Saving before quit...")
            self.save(label)
            raise SystemExit

    def save(self, label=''):
        os.makedirs('models', exist_ok=True)
        save_filename = os.path.join('models', os.path.splitext(os.path.basename(self.filename))[0] + label + '.pt')
        torch.save(self.decoder, save_filename)
        print('Saved as %s' % save_filename)


# Run as standalone script
if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    # argparser.add_argument('filename', type=str)
    argparser.add_argument('--model', type=str, default="gru")
    argparser.add_argument('--n_epochs', type=int, default=2000)
    argparser.add_argument('--print_every', type=int, default=1)
    argparser.add_argument('--hidden_size', type=int, default=100)
    argparser.add_argument('--n_layers', type=int, default=2)
    argparser.add_argument('--learning_rate', type=float, default=0.01)
    argparser.add_argument('--chunk_len', type=int, default=200)
    argparser.add_argument('--batch_size', type=int, default=100)
    argparser.add_argument('--nocuda', action='store_true', default=False)
    args = argparser.parse_args()

    print('Running tests')
    for model in ["gru", "lstm"]:
        for hidden_size in [100, 50]:
            for lr in [0.003, 0.01, 0.03, 0.1, 0.3][::-1]:
                t = Trainer(filename='shakespeare.txt', model=model, hidden_size=hidden_size, cuda=not args.nocuda)
                t.train(lr, n_epochs=200)
