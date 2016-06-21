import numpy as np

import chainer.functions as F
from chainer import Variable

from chainn import functions as UF
from chainn.model import EnsembleModel
from chainn.util import DecodingOutput
from chainn.classifier import RNN
from chainn.model.text import RecurrentLSTMLM

class RNNLM(RNN):
    def __init__(self, *args, operation=None, **kwargs):
        self._all_models = [RecurrentLSTMLM]
        self._operation = operation
        super(RNN, self).__init__(*args, **kwargs)
    
    def _collect_decoding_output(self, time_step, holder, out, word=None, src=None):
        if self._operation == "gen":
            y = UF.argmax(out.y.data)
            for i in range(len(y)):
                holder.y[i][time_step] = y[i]
        else:
            loss = 1 - out.y.data[0][src[time_step]]
            holder.loss += loss 
    
    def _init_output(self, batch_size, gen_limit, src_len):
        return DecodingOutput({ "y": np.zeros((batch_size, gen_limit), dtype=np.float32), "loss": 0})

    def get_vocabularies(self):
        if type(self._model) == EnsembleModel:
            return self._model[0]._src_voc
        else:
            return self._model._src_voc
 
