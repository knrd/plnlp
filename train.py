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
        # this will be setup in train method
        self.decoder_optimizer = None

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

    def train_step(self, inp, target):
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
        self.decoder_optimizer.step()

        _, argmax = torch.max(output, 1)
        accuracy = (target[:, c] == argmax).sum().item() / argmax.size(0)
        chunk_loss = loss.item() / self.chunk_len

        return chunk_loss, accuracy

    def train(self, learning_rate, n_epochs=200, skip_if_tested=False, save_model=True, train_saved_model=None, save_logs=True, version=0, checkpoint_every=0):
        file_name = os.path.splitext(os.path.basename(self.content_reader.file_name))[0]
        label = "%s--e_%d-m_%s-lr_%s-hs_%d" % (file_name, n_epochs, self.model, learning_rate, self.hidden_size)
        if version:
            label += "_ver-%s" % version

        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate)
        epoch = 0
        epoch_delta = 0
        loss = 0
        if train_saved_model:
            if not version:
                raise AttributeError('provide version')
            epoch_delta = self.load_saved_model(train_saved_model)

        if skip_if_tested:
            if not self.logger:
                raise AttributeError('logger is not provided')
            if os.path.exists(self.logger.get_path(label)):
                print("Test %s exists, skipping" % label, flush=True)
                return None

        epoch_to_save = epoch + epoch_delta
        self.decoder.train()
        try:
            print("Training %s for %d epochs..." % (label, n_epochs), flush=True)
            tt = tqdm(range(1, n_epochs + 1))
            for epoch in tt:
                epoch_to_save = epoch + epoch_delta
                inp, target = self._random_training_set()
                loss, accuracy = self.train_step(inp, target)
                tt.set_description('loss=%g' % loss)
                self.loss_avg += loss

                if self.logger and save_logs:
                    info = {'loss': loss, 'accuracy': accuracy, 'learning_rate': learning_rate}
                    for tag, value in info.items():
                        self.logger.scalar_summary(label, tag, value, epoch_to_save)

                if checkpoint_every and epoch % checkpoint_every == 0:
                    self.save_model(label, epoch_to_save, loss, checkpoint=epoch)
                    print('Saved model checkpoint %s at %d' % (label, epoch), flush=True)

            if save_model:
                self.save_model(label, epoch_to_save, loss)
                print('Saved model %s' % label, flush=True)

        except KeyboardInterrupt:
            if save_model:
                print("Saving before quit...", flush=True)
                self.save_model(label, epoch_to_save, loss)
                raise SystemExit

        if self.logger and save_logs:
            self.logger.flush(label)

    def save_model(self, label, epoch, loss, checkpoint=0):
        save_filename = self.get_model_path(label)
        if checkpoint:
            save_filename += "check%d" % checkpoint
        cpu_model = self.get_cpu_decoder_copy()
        torch.save({
            'model_state': cpu_model.state_dict(),
            'optimizer_state': self.decoder_optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'checkpoint': checkpoint,
            'version': (0, 1),
            'model_hidden_size': self.hidden_size,
            'model_type': self.model,
            'model_n_layers': self.n_layers,
            'model_char_dict': self.content_reader.char_dict
        }, save_filename)
        torch.save(cpu_model, save_filename + "cpu")

    def get_model_path(self, label):
        os.makedirs('models', exist_ok=True)
        return os.path.join('models', label + '.pt')

    def get_cpu_decoder_copy(self):
        return copy.deepcopy(self.decoder).cpu()

    def load_saved_model(self, file_name):
        saved_model_path = self.get_model_path(file_name)
        if os.path.isfile(saved_model_path):
            state = torch.load(saved_model_path)
            self.decoder.load_state_dict(state['model_state'])
            self.decoder_optimizer.load_state_dict(state['optimizer_state'])
            if self.cuda:
                self.decoder.cuda()
            print("Loaded saved model", saved_model_path, flush=True)
            return state['epoch']
        else:
            print("Saved model", saved_model_path, " does not exists. Exiting", flush=True)
            raise SystemExit


# Run as standalone script
if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('--model', type=str, default="lstm")
    argparser.add_argument('--n_epochs', type=int, default=5)
    argparser.add_argument('--hidden_size', type=int, default=512)
    # argparser.add_argument('--n_layers', type=int, default=2)
    argparser.add_argument('--learning_rate', type=float, default=0.003)
    # argparser.add_argument('--chunk_len', type=int, default=200)
    # argparser.add_argument('--batch_size', type=int, default=100)
    argparser.add_argument('--cuda', action='store_true', default=False)
    args = argparser.parse_args()

    # content = FileReader('content', 'psy.txt')
    content = FileReader('content', args.filename)
    print(len(content.char_dict), content.char_dict[1:])

    print('Running tests', 'using CUDA' if args.cuda else 'CPU', flush=True)
    t = Trainer(content_reader=content, model=args.model, hidden_size=args.hidden_size, cuda=args.cuda)
    # t.train(args.learning_rate, n_epochs=args.n_epochs, train_saved_model='przygody-tomka-sawyera--e_3-m_lstm-lr_0.01-hs_32', version=2, checkpoint_every=1)
    t.train(args.learning_rate, n_epochs=args.n_epochs)

    generated_text = generate(t.get_cpu_decoder_copy(), prime_str='Olo Angela.')
    print(generated_text, '\n', flush=True)
