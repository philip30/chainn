
import sys
from chainn.util import functions as UF
from chainn.util.io import batch_generator

class Tester:
    def __init__(self, params, loader):
        self.assigned_gpu      = UF.init_global_environment(0, params.gpu, params.use_cpu)
        self.decoding_options  = self.load_decoding_options(params)
        self.loader            = loader
        self.classifier        = self.load_classifier(params)
   
    # Predict output
    def test(self, stream=sys.stdin):
        self.onDecodingStart()
        for i, line in enumerate(stream):
            inp = list(batch_generator(\
                    self.loader([line.strip()], self._src_voc), (self._src_voc,), 1))[0][0]
            out = self.classifier.classify(inp, **self.decoding_options)
            self.onSingleUpdate(i, inp, out)
        self.onDecodingFinish()
    
    def onDecodingStart(self):
        UF.trace("Decoding started.")

    def onDecodingFinish(self):
        pass

    # Abstract Methods
    def onSingleUpdate(self, ctr, src, trg):
        raise NotImplementedError()

    def load_decoding_options(self, params):
        raise NotImplementedError()

    def load_classifier(self, params):
        raise NotImplementedError()
    
