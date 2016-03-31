
import sys
from chainn.util import functions as UF
from chainn.util.io import batch_generator

class Tester:
    def __init__(self, loader, inp_vocab, onDecodingStart, onBatchUpdate, onSingleUpdate, onDecodingFinish, out_vocab=None, options={}, batch=1):
        self._inp_vocab        = inp_vocab
        self._out_vocab        = out_vocab
        self._decoding_options = options
        self._batch            = batch
        self.loader            = loader
        self.onDecodingStart   = onDecodingStart
        self.onBatchUpdate     = onBatchUpdate
        self.onSingleUpdate    = onSingleUpdate
        self.onDecodingFinish  = onDecodingFinish
   
    def batched_decoding(self, data, model):
        # Load data
        with open(data) as inp_fp:
            data = self.loader(inp_fp, self._inp_vocab)
        
        self.onDecodingStart()

        # Start Decoding
        output = {}
        data   = batch_generator(data, (self._inp_vocab,), batch_size=self._batch)
        ctr    = 0
        for src, src_id in data:
            trg = model.classify(src, **self._decoding_options)
            
            # Collecting output
            for src_i, trg_i, id_i in zip(src, trg, src_id):
                output[id_i] = src_i, trg_i
           
            self.onBatchUpdate(ctr, src, trg)
            ctr += len(src)
        
        self.onDecodingFinish(data, output)
        
    def single_decoding(self, data, model):
        self.onDecodingStart()
        for i, line in enumerate(data):
            inp = list(batch_generator(self.loader([line.strip()], self._inp_vocab), (self._inp_vocab,), 1))[0][0]
            out = model.classify(inp, **self._decoding_options)

            self.onSingleUpdate(i, inp, out)

    def test(self, inp, model):
        if inp:
            return self.batched_decoding(inp, model)
        else:
            return self.single_decoding(sys.stdin, model)
    
    def eval(self, data, model):
        self.onDecodingStart()

        # Start Decoding
        ctr        = 0
        accum_loss = 0
        for src, trg in data:
            loss = model.eval(src, trg, **self._decoding_options)
            
            self.onSingleUpdate(ctr, src, loss)
            accum_loss += loss
            ctr += len(src)
        
        self.onDecodingFinish(data, accum_loss / len(data))

