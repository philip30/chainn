from collections import defaultdict

UNK = "<UNK>"
EOS = "<EOS>"

class Vocabulary(object):
    # Overloading methods
    def __init__(self, unk=True, eos=True):
        self._data = defaultdict(lambda: len(self._data))
        self._back = {}

        if unk:
            self[UNK]

        if eos:
            self[EOS]

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

    def __equal__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self._data == other._data and self._back == other._back

    # Public
    def str_rpr(self, data):
        ret = []
        for tok in data:
            ret.append(self.tok_rpr(tok))
            ret.append(EOS)
            ret = ret[:ret.index(EOS)]
        return " ".join(ret)

    def tok_rpr(self, wid):
        if wid in self._back:
            return self._back[wid]
        else:
            return UNK

    def save(self, fp):
        fp.write(len(self._data))
        for word, index in sorted(self._data.items(), key=lambda x:x[0]):
            fp.write(str(index) + "\t" + str(word))
    
    def unk_id(self):
        return self[UNK]

    def eos_id(self):
        return self[EOS]
    
    def unk(self):
        return UNK

    def eos(self):
        return EOS

    @staticmethod
    def load(fp):
        size = int(fp.read())
        self = Vocabulary()
        for i in range(size):
            index, word = fp.read().strip().split("\t")
            self[word] = int(index)
        return self

