import numpy as np

import chainer.functions as F
from chainer import Variable

from chainn import functions as UF
from chainn.model import ChainnClassifier
from chainn.model.basic import RNN, LSTMRNN
from chainn.util.io import ModelFile

class ParallelTextClassifier(ChainnClassifier):
    
    def __init__(self, *args, **kwargs):
        self._all_models = [RNN, LSTMRNN]
        super(ParallelTextClassifier, self).__init__(*args, **kwargs)
    
    def __call__(self, x_data, y_data=None, is_train=True):
        xp         = self._xp
        batch_size = len(x_data)
        src_len    = len(x_data[0])
        model      = self._model
    
        accum_loss = 0
        accum_acc  = 0
        
        # Forward Computation
        model.predictor.reset_state(batch_size)
        output = [[] for _ in range(batch_size)]
        
        # For each word
        for j in range(src_len):
            words  = Variable(xp.array([x_data[i][j] for i in range(batch_size)], dtype=np.int32))
           
            if y_data is not None:
                labels = Variable(xp.array([y_data[i][j] for i in range(batch_size)], dtype=np.int32))
                accum_loss += model(words, labels)
            
            if not y_data is not None or self._collect_output:
                y = UF.argmax(model.y.data if y_data is not None else model.predictor(words).data)
                for i in range(len(y)):
                    output[i].append(y[i])
        
        if y_data is not None:
            accum_loss = accum_loss / src_len
            return accum_loss, output
        else:
            return output

        
