# Author: philip arthur (philip.arthur30@gmail.com)
# Description:
#   Class to represent vocabulary word id. It supports the back mapping.
# Test: test/test_vocabulary.py

UNK = "<UNK>"
EOS = "<EOS>"
STUFF = "{*}"

class Vocabulary(object):
    def __init__(self, add_unk=False, add_eos=False, add_stuff=False):
        self.word_to_id = {}
        self.id_to_word = {}

        if add_unk:
            self.add_word(UNK)

        if add_eos:
            self.add_word(EOS)

        if add_stuff:
            self.add_word(STUFF)

    def add_word(self, word):
        word_id = self.word_to_id.get(word, len(self.word_to_id))
        self.word_to_id[word]    = word_id
        self.id_to_word[word_id] = word
        return word_id

    def set_word(self, word, word_id):
        self.word_to_id[word]    = word_id
        self.id_to_word[word_id] = word

    def __getitem__(self, word):
        return self.word_to_id[word]

    def __contains__(self, word):
        return word in self.word_to_id

    def __iter__(self):
        return iter(self.word_to_id)

    def __len__(self):
        return len(self.word_to_id)

    def __reversed__(self):
        return reversed(self.word_to_id)

    def __str__(self):
        return str(self.word_to_id)

    def __equal__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self.id_to_word == other.id_to_word and \
                    self.word_to_id == other.word_to_id

    # Public
    def sentence(self, word_ids, append_eos = True):
        ret = []
        for word_id in word_ids:
            ret.append(self.word(word_id))
        if append_eos:
            ret.append(EOS)
            ret = ret[:ret.index(EOS)]
        return " ".join(ret)

    def word(self, word_id):
        return self.id_to_word.get(word_id, UNK)

    def unk_id(self):
        return self[UNK]

    def eos_id(self):
        return self[EOS]

    def stuff_id(self):
        return self[STUFF]

    def unk(self):
        return UNK

    def eos(self):
        return EOS

