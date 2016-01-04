import numpy as np
from collections import defaultdict

# Chainer
from chainer import Variable
import chainer.functions as F

# Chainn
import chainn.util.functions as UF
from chainn.model import ChainnClassifier
from chainn.model.nmt import EncoderDecoder, Attentional
from chainn.util import ModelFile
from chainn.link import NMTClassifier

class EncDecNMT(ChainnClassifier):

    def __init__(self, *args, **kwargs):
        self._all_models = [EncoderDecoder, Attentional]
        super(EncDecNMT, self).__init__(*args, **kwargs)
        
    def __call__(self, x_data, y_data=None, gen_limit=50):
        # Unpacking
        xp         = self._xp
        batch_size = len(x_data)
        model      = self._model
        is_train   = y_data is not None
        EOL        = self._trg_voc.eos_id()
        
        # If it is training, gen limit should be = y_data
        if y_data is not None:
            gen_limit = len(y_data[0])
        # Perform encoding + Reset state
        model.predictor.reset_state(x_data)

        output = [[] for _ in range(batch_size)]
        accum_loss, accum_acc = 0, 0 

        # Decoding
        for j in range(gen_limit):
            if is_train:
                s_t = Variable(xp.array([y_data[i][j] for i in range(len(y_data))], dtype=np.int32))
                accum_loss += self._model(x_data, s_t) # Decode one step
                accum_acc  += model.accuracy
            
            # Collecting output
            if not is_train or self._collect_output:
                y = UF.argmax(model.y.data if is_train else model.predictor(x_data).data)
                for i in range(len(y)):
                    output[i].append(y[i])
            
            # Break if all sentences end with EOL
            if not is_train and all(output[i][j] == EOL for i in range(len(output))):
                break

        if is_train:
            accum_loss = accum_loss / batch_size
            accum_acc  = accum_acc  / batch_size
            return accum_loss, accum_acc, output
        else:
            return output
    
    def _load_classifier(self):
        return NMTClassifier

