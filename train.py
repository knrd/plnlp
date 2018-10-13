#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import argparse
import copy
import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm

from content_reader import FileReader
from generate import generate
from model import CharRNN2
from tblogger import TBLogger


class Trainer(object):
    def __init__(self, content_reader, model="gru", hidden_size=100, n_layers=2, cuda=True, chunk_len=200, batch_size=100, tensorboard=True, verbose=1):
        self.verbose = verbose
        self.cuda = cuda
        self.content_reader = content_reader
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
        self.decoder = CharRNN2(
            hidden_size=self.hidden_size,
            model=self.model,
            n_layers=self.n_layers,
            char_dict=content_reader.char_dict
        )
        self.criterion = nn.CrossEntropyLoss()

        if self.cuda:
            self.decoder.cuda()

    def _random_training_set(self):
        inp = torch.LongTensor(self.batch_size, self.chunk_len)
        target = torch.LongTensor(self.batch_size, self.chunk_len)
        for bi in range(self.batch_size):
            start_index = random.randint(0, self.content_reader.content_len - self.chunk_len - 1)
            end_index = start_index + self.chunk_len + 1
            chunk = self.content_reader.content[start_index:end_index]
            inp[bi] = self.decoder.char2tensor(chunk[:-1])
            target[bi] = self.decoder.char2tensor(chunk[1:])

            # assert (inp[bi] == t_inp).all().item()
            # assert (target[bi] == t_target).all().item()

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
        accuracy = (target[:, c] == argmax).sum().item() / argmax.size(0)
        chunk_loss = loss.item() / self.chunk_len

        return chunk_loss, accuracy

    def train(self, learning_rate, n_epochs=200, skip_if_tested=False, save_model=True, train_saved_model=None, save_logs=True, version=0):
        file_name = os.path.splitext(os.path.basename(self.content_reader.file_name))[0]
        label = "%s--e_%d-m_%s-lr_%s-hs_%d-nl-%d" % (file_name, n_epochs, self.model, learning_rate, self.hidden_size, self.n_layers)
        if version:
            label += "_ver-%s" % version

        if train_saved_model:
            self.load_saved_model(train_saved_model)

        if skip_if_tested:
            if not self.logger:
                raise AttributeError('logger is not provided')
            if os.path.exists(self.logger.get_path(label)):
                print("Test %s exists, skipping" % label, flush=True)
                return None

        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate)
        try:
            print("Training %s for %d epochs..." % (label, n_epochs), flush=True)
            tt = tqdm(range(1, n_epochs + 1))
            for epoch in tt:
                inp, target = self._random_training_set()
                loss, accuracy = self.train_step(inp, target, decoder_optimizer)
                tt.set_description('loss=%g' % loss)
                self.loss_avg += loss

                if self.logger and save_logs:
                    # 1. Log scalar values (scalar summary)
                    info = {'loss': loss, 'accuracy': accuracy, 'learning_rate': learning_rate}

                    for tag, value in info.items():
                        self.logger.scalar_summary(label, tag, value, epoch)

            if save_model:
                self.save(label)

        except KeyboardInterrupt:
            if save_model:
                print("Saving before quit...", flush=True)
                self.save(label)
                raise SystemExit

    def save(self, label):
        save_filename = self.get_model_path(label)
        cpu_model = self.get_cpu_decoder_copy()
        torch.save(cpu_model, save_filename)
        self.logger.flush(label)
        print('Saved as %s' % save_filename, flush=True)

    def get_model_path(self, label):
        os.makedirs('models', exist_ok=True)
        return os.path.join('models', label + '.pt')

    def get_cpu_decoder_copy(self):
        return copy.deepcopy(self.decoder).cpu()

    def load_saved_model(self, file_name):
        saved_model_path = self.get_model_path(file_name)
        if os.path.isfile(saved_model_path):
            self.decoder = torch.load(saved_model_path)
            if self.cuda:
                self.decoder.cuda()
            print("Loading saved model", saved_model_path, flush=True)
        else:
            print("Saved model", saved_model_path, " does not exists. Exiting", flush=True)
            raise SystemExit


# Run as standalone script
if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    # argparser.add_argument('filename', type=str)
    # argparser.add_argument('--model', type=str, default="gru")
    # argparser.add_argument('--n_epochs', type=int, default=2000)
    # argparser.add_argument('--print_every', type=int, default=1)
    # argparser.add_argument('--hidden_size', type=int, default=100)
    # argparser.add_argument('--n_layers', type=int, default=2)
    # argparser.add_argument('--learning_rate', type=float, default=0.01)
    # argparser.add_argument('--chunk_len', type=int, default=200)
    # argparser.add_argument('--batch_size', type=int, default=100)
    argparser.add_argument('--nocuda', action='store_true', default=False)
    args = argparser.parse_args()

    use_cuda = 1 and not args.nocuda

    # content = FileReader('content', 'psy.txt')
    content = FileReader('content', 'shakespeare.txt')
    print(len(content.char_dict), content.char_dict[1:])

    print('Running tests', 'using CUDA' if use_cuda else 'CPU', flush=True)
    for model in ["lstm"]:
        for hidden_size in [256]:
            for lr in [0.01][::-1]:
                t = Trainer(content_reader=content, model=model, hidden_size=hidden_size, cuda=use_cuda)
                t.train(lr, n_epochs=10, save_model=False, save_logs=False)

                generated_text = generate(t.get_cpu_decoder_copy(), prime_str='Olo Angela.')
                print(generated_text, '\n', flush=True)
