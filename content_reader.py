import os


class FileReader(object):
    def __init__(self, folder, file_name):
        self.file_name = file_name
        self.path = os.path.join(folder, file_name)

        with open(self.path, 'r') as f:
            self.content = f.read()
        self.content_len = len(self.content)
        char_list = sorted(set(self.content))
        self.char_dict = "\0" + "".join(char_list)
