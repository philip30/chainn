import numpy as np
import math
from collections import defaultdict

# Chainer
from chainer import Variable, cuda
import chainer.functions as F

# Chainn
import chainn.util.functions as UF
from chainn.classifier import ChainnClassifier
from chainn.model.nmt import EncoderDecoder, Attentional, DictAttentional
from chainn.util import DecodingOutput

class EncDecNMT(ChainnClassifier):
    def __init__(self, *args, **kwargs):
        self._all_models = [EncoderDecoder, Attentional, DictAttentional]
        super(EncDecNMT, self).__init__(*args, **kwargs)

    def _init_output(self, batch_size, gen_limit, src_len):
        y = np.zeros((batch_size, gen_limit), dtype=np.float32)
        a = np.zeros((batch_size, gen_limit, src_len), dtype=np.float32)
        return DecodingOutput({ "y": y, "a": a })
    
    def _collect_decoding_output(self, src_col, holder, out, word=None, src=None):
        if word is None:
            y = UF.argmax(out.y.data)
            for i in range(len(y)):
                holder.y[i][src_col] = y[i]
        else:
            holder.y[0][src_col] = word
        
        try:
            a = out.a
            if a is not None:
                for i, x in enumerate(a.data):
                    for j, x_a in enumerate(x):
                        holder.a[i][src_col][j] = float(x_a)
        except:
            holder.a = None
    
    def _prepare_decode_input(self, input_batch, time_step, is_train=True):
        return input_batch

