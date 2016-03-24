import numpy as np
from collections import defaultdict

# Chainer
from chainer import Variable
import chainer.functions as F

# Chainn
import chainn.util.functions as UF
from chainn.model import ChainnClassifier
from chainn.model.nmt import EncoderDecoder, Attentional, DictAttentional
from chainn.util import DecodingOutput
from chainn.util.io import ModelFile
from chainn.link import NMTClassifier

# Length of truncated BPTT
BP_LEN = 1000

class EncDecNMT(ChainnClassifier):

    def __init__(self, *args, **kwargs):
        self._all_models = [EncoderDecoder, Attentional, DictAttentional]
        super(EncDecNMT, self).__init__(*args, **kwargs)
        
    def __call__(self, x_data, y_data=None, is_train=True, gen_limit=50, *args, **kwargs):
        # Unpacking
        xp         = self._xp
        batch_size = len(x_data)
        model      = self._model
        EOL        = self._trg_voc.eos_id()
        
        # If it is training, gen limit should be = y_data
        if y_data is not None:
            gen_limit = len(y_data[0])
        # Perform encoding + Reset state
        model.predictor.reset_state(x_data, y_data, is_train, *args, **kwargs)

        output    = [[] for _ in range(batch_size)]
        alignment = [[[] for _ in range(gen_limit)] for _ in range(batch_size)]
        accum_loss = 0
        bp_ctr = 0

        # Decoding
        for j in range(gen_limit):
            if y_data is not None:
                s_t = Variable(xp.array([y_data[i][j] for i in range(len(y_data))], dtype=np.int32))
                accum_loss += model(x_data, s_t, is_train=is_train, *args, **kwargs) # Decode one step
                out = model.output
            else:
                out = model(x_data, is_train=is_train, *args, **kwargs)
            
            # Collecting output
            if not is_train or self._collect_output:
                y = UF.argmax(out.y.data)
                for i in range(len(y)):
                    output[i].append(y[i])
                
                a = out.a
                if a is not None:
                    for i, x in enumerate(a.data):
                        for x_a in x:
                            alignment[i][j].append(float(x_a))
                        # Break if all sentences end with EOL
                else:
                    alignment = None
            if not is_train and all(output[i][j] == EOL for i in range(len(output))):
                break

            if y_data is not None:
                bp_ctr += 1
                if bp_ctr % BP_LEN == 0:
                    bp_ctr = 0
                    accum_loss.backward()
                    accum_loss.unchain_backward()

        output = DecodingOutput(output, alignment)
        if y_data is not None:
            accum_loss = accum_loss / gen_limit
            return accum_loss, output
        else:
            return output
    
    def _load_classifier(self):
        return NMTClassifier
    
    def report(self):
        self._model.predictor.report()

