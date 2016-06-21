
import sys
from chainn.util import functions as UF
from chainn.util.io import batch_generator

class Tester:
    def __init__(self, loader, inp_vocab, onDecodingStart, onSingleUpdate, onDecodingFinish, out_vocab=None, options={}):
        self._inp_vocab        = inp_vocab
        self._out_vocab        = out_vocab
        self._decoding_options = options
        self.loader            = loader
        self.onDecodingStart   = onDecodingStart
        self.onSingleUpdate    = onSingleUpdate
        self.onDecodingFinish  = onDecodingFinish
   
    # Predict output
    def test(self, classifier):
        return self.__single_decoding(sys.stdin, classifier)
       
    ### Local Routines
    def __single_decoding(self, data, classifier):
        self.onDecodingStart()
        for i, line in enumerate(data):
            inp = list(batch_generator(self.loader([line.strip()], self._inp_vocab), (self._inp_vocab,), 1))[0][0]
            out = classifier.classify(inp, **self._decoding_options)
            self.onSingleUpdate(i, inp, out)
        self.onDecodingFinish()

