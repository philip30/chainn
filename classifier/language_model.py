import numpy as np

import chainer.functions as F
from chainer import Variable

from chainn import functions as UF
from chainn.classifier import ParallelTextClassifier
from chainn.model.text import RecurrentLSTMLM
from chainn.util.io import ModelFile

def collect_output(src_col, output, out):
    for i in range(len(out)):
        output[i][src_col] = out[i]

class LanguageModel(ParallelTextClassifier):

    def __init__(self, *args, **kwargs):
        self._all_models = [RecurrentLSTMLM]
        super(ParallelTextClassifier, self).__init__(*args, **kwargs)
    
    def _classify(self, x_data, gen_limit=50, *args, **kwargs):
        xp         = self._xp
        batch_size = len(x_data)
        src_len    = len(x_data[0])
        EOL        = self._model._trg_voc.eos_id()
 
        self._model.reset_state(batch_size)
        output     = np.zeros((batch_size, gen_limit), dtype=np.float32)
        
        # For each word
        for j in range(min(gen_limit, src_len)):
            words  = Variable(xp.array([x_data[i][j] for i in range(batch_size)], dtype=np.int32))
            collect_output(j, output, words.data)
            y = self._model(words, is_train=False)
            
        start_gen = j
        for j in range(start_gen, gen_limit - src_len):
            next_word = UF.argmax(y.data)
            collect_output(j, output, next_word)
            y = self._model(Variable(xp.array(next_word, dtype=np.int32)), is_train=False)
            if all(word == EOL for word in next_word):
                break

        return output

