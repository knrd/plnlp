import torch


class FileReader(object):
    def __init__(self, file_name):
        self.file_name = file_name

        with open(self.file_name) as f:
            self.content = f.read()
        self.content_len = len(self.content)
        char_list = sorted(set(self.content))
        self.char_dict = "\0" + "".join(char_list)
        self.char_dict_len = len(self.char_dict)

    # Turning a string into a tensor
    def char2tensor(self, s):
        i_in = [self.char_dict.find(s[i]) for i in range(0, len(s))]
        # replacing -1 with 0
        i_in = [i if i >= 0 else 0 for i in i_in]

        return torch.LongTensor(i_in)
