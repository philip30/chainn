import numpy as np

import chainer.functions as F
from chainer import Variable

from chainn import functions as UF
from chainn.classifier import ChainnClassifier
from chainn.model.text import RecurrentLSTM
from chainn.util.io import ModelFile

def collect_output(src_col, output, out):
    y = UF.argmax(out.data)
    for i in range(len(y)):
        output[i][src_col] = y[i]

class ParallelTextClassifier(ChainnClassifier):
    def __init__(self, *args, **kwargs):
        self._all_models = [RecurrentLSTM]
        super(ParallelTextClassifier, self).__init__(*args, **kwargs)
   
    def _train(self, x_data, y_data, is_dev, *args, **kwargs):
        xp         = self._xp
        batch_size = len(x_data)
        src_len    = len(x_data[0])
        is_train   = not is_dev
   
        self._model.reset_state(batch_size, is_train=is_train)
        output     = None
        if self._collect_output:
            output = np.zeros((batch_size, src_len), dtype=np.float32)
        
        # For each word
        accum_loss = 0
        for j in range(src_len):
            words  = Variable(xp.array([x_data[i][j] for i in range(batch_size)], dtype=np.int32))
            labels = Variable(xp.array([y_data[i][j] for i in range(batch_size)], dtype=np.int32))
            y      = self._model(words, labels, is_train=is_train)
            accum_loss += self._calculate_loss(y, labels, is_train)
            
            if self._collect_output:
                collect_output(j, output, y)
        
        return (accum_loss / src_len), output

    def _classify(self, x_data, *args, **kwargs):
        xp         = self._xp
        batch_size = len(x_data)
        src_len    = len(x_data[0])
 
        self._model.reset_state(batch_size, is_train=False)
        output     = np.zeros((batch_size, src_len), dtype=np.float32)
        
        # For each word
        for j in range(src_len):
            words  = Variable(xp.array([x_data[i][j] for i in range(batch_size)], dtype=np.int32))
            y      = self._model(words, is_train=False)
            collect_output(j, output, y)

        return output
    
    def _eval(self, x_data, y_data, *args, **kwargs):
        xp         = self._xp
        batch_size = len(x_data)
        src_len    = len(x_data[0])
 
        self._model.reset_state(batch_size)
        accum_loss = 0

        # For each word
        for j in range(src_len):
            words  = Variable(xp.array([x_data[i][j] for i in range(batch_size)], dtype=np.int32))
            labels = Variable(xp.array([y_data[i][j] for i in range(batch_size)], dtype=np.int32))
            y      = self._model(words, labels)
            accum_loss += self._calculate_loss(y, labels, False)

        return accum_loss / src_len

