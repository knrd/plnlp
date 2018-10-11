# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import time
import math
import torch

# Reading and un-unicode-encoding data

all_characters = string.printable
n_characters = len(all_characters)


def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


# Turning a string into a tensor
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor


def start_time():
    return time.time()


# Readable time elapsed
def time_since(since):
    s = start_time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
