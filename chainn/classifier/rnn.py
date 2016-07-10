import numpy as np

import chainer.functions as F
from chainer import Variable

from chainn import functions as UF
from chainn.util import DecodingOutput
from chainn.classifier import ChainnClassifier
from chainn.model.text import RecurrentLSTM

class RNN(ChainnClassifier):
    def __init__(self, *args, **kwargs):
        self._all_models = [RecurrentLSTM]
        super(RNN, self).__init__(*args, **kwargs)

    def _init_output(self, batch_size, gen_limit, src_len):
        return DecodingOutput({ "y": np.zeros((batch_size, gen_limit), dtype=np.float32) })

    def _collect_decoding_output(self, time_step, holder, out, word=None, src=None):
        if word is None:
            y = UF.argmax(out.y.data)
            for i in range(len(y)):
                holder.y[i][time_step] = y[i]
        else:
            holder.y[0][time_step] = word

    def _prepare_decode_input(self, input_batch, time_step, is_train=True):
        volatile = "off" if is_train else "on"
        if time_step >= len(input_batch[0]):
            return None
        return Variable(self._xp.array([input_batch[i][time_step] for i in range(len(input_batch))], dtype=np.int32), volatile=volatile)

