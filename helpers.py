# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import time
import math
import torch

# Reading and un-unicode-encoding data

all_characters = "\0" + string.printable
n_characters = len(all_characters)


def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


# Turning a string into a tensor
def char2tensor(s):
    i_in = [all_characters.find(s[i]) for i in range(0, len(s) - 1)]
    i_out = [all_characters.find(s[i]) for i in range(1, len(s))]

    # replacing -1 with 0
    i_in = [i if i >= 0 else 0 for i in i_in]
    i_out = [i if i >= 0 else 0 for i in i_out]

    t_in = torch.tensor(i_in, dtype=torch.int64)
    t_out = torch.tensor(i_out, dtype=torch.int64)
    return t_in, t_out


def start_time():
    return time.time()


# Readable time elapsed
def time_since(since):
    s = start_time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
