import torch


class FileReader(object):
    def __init__(self, file_name):
        self.file_name = file_name

        with open(self.file_name) as f:
            self.file_content = f.read()
        self.file_content_len = len(self.file_content)
        char_list = sorted(set(self.file_content))
        self.char_dict = "\0" + "".join(char_list)
        self.char_dict_len = len(self.char_dict)

    # Turning a string into a tensor
    def char2tensor(self, s):
        i_in = [self.char_dict.find(s[i]) for i in range(0, len(s) - 1)]
        i_out = [self.char_dict.find(s[i]) for i in range(1, len(s))]

        # replacing -1 with 0
        i_in = [i if i >= 0 else 0 for i in i_in]
        i_out = [i if i >= 0 else 0 for i in i_out]

        t_in = torch.tensor(i_in, dtype=torch.int64)
        t_out = torch.tensor(i_out, dtype=torch.int64)
        return t_in, t_out
