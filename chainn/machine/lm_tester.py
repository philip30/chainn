import math
from chainn.classifier import RNNLM
from chainn.machine import Tester
from chainn.util import functions as UF
from chainn.util.io import load_lm_gen_data

class LMTester(Tester):
    def __init__(self, params):
        super(LMTester, self).__init__(params, load_lm_gen_data)
        self.operation = params.operation
        self.total_loss = 0
        self.total_sent = 0

    def load_classifier(self, params):
        model = RNNLM(params, use_gpu=self.assigned_gpu, collect_output=True, operation=params.operation)
        vocab = model.get_vocabularies()
        self._src_voc = vocab
        self._trg_voc = vocab
        return model
    
    def load_decoding_options(self, params):
        return { "beam": params.beam, "eos_disc": params.eos_disc }

    def onSingleUpdate(self, ctr, src, trg):
        if self.operation == "gen":
            print(self._trg_voc.str_rpr(trg.y[0]))
        else:
            loss       = trg.loss / len(src[0])
            self.total_loss += loss
            self.total_sent += 1
            if self.operation == "sppl":
                print(math.exp(loss))

    def onDecodingStart(self):
        op = self.operation
        if op == "gen":
            UF.trace("Sentence generation started.")
        elif op == "cppl":
            UF.trace("Corpus PPL calculation started.")
        elif op == "sppl":
            UF.trace("Sentence PPL calculation started.")
        else:
            raise ValueError(op)

    def onDecodingFinish(self):
        if self.operation == "cppl":
            print(math.exp(float(self.total_loss)/self.total_sent))

