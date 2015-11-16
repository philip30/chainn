from collections import defaultdict

UNK = "<UNK>"
class Vocabulary:
    # Overloading methods
    def __init__(self):
        self._data = defaultdict(lambda: len(self._data))
        self._back = {}

    def __getitem__(self, index):
        id = self._data[index]
        if id not in self._back:
            self._back[id] = index
        return id

    def __setitem__(self, index, data_i):
        self._data[index] = data_i
        self._back[data_i] = index

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __reversed__(self):
        return reversed(self._data)

    def __str__(self):
        return str(self._data)

    # Public
    def str_rpr(self, data, EOS=None):
        ret = []
        for tok in data:
            ret.append(self.tok_rpr(tok))
        if EOS is not None:
            ret.append(EOS)
            ret = ret[:ret.index(EOS)]
        return " ".join(ret)

    def tok_rpr(self, wid):
        if wid in self._back:
            return self._back[wid]
        else:
            return UNK

    def fill_from_file(self, file):
        with open(file, "r") as f:
            for line in f:
                line = line.strip().lower().split()
                for tok in line:
                    self.__getitem__(tok)
