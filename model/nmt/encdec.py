import numpy as np
import chainer.links as L
import chainer.functions as F

# Chainer
from chainer import Variable

# Chainn
from chainn import functions as UF
from chainn.model.basic import ChainnBasicModel
from chainn.util import DecodingOutput
from chainn.link import StackLSTM

# By Philip Arthur (philip.arthur30@gmail.com)
# This program is an implementation of Sequence to Sequence Learning with Neural Networks
# (Sutskever et al., 2013)
# http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

class EncoderDecoder(ChainnBasicModel):
    name = "encdec"    
    
    def _construct_model(self, input, output, hidden, depth, embed):
        assert(depth >= 1)
        I, O, E, H = input, output, embed, hidden
        
        self.IE = L.EmbedID(I,E)
        self.EF = StackLSTM(E,H,depth)
        self.WS = L.Linear(H, O)
        self.OE = L.EmbedID(O, E)
        self.HH = L.Linear(H, H)

        ret = []
        # Encoder
        ret.append(self.IE)
        ret.append(self.EF)
        ret.append(self.WS)
        ret.append(self.OE)
        ret.append(self.HH)

        return ret
    
    # Encoding all the source sentence
    def reset_state(self, x_data, y_data, *args, **kwargs):
        # Unpacking
        batch_size = len(x_data)
        src_len    = len(x_data[0])
        xp = self._xp
        f  = self._activation
        is_train = y_data is not None
        self.EF.reset_state()

        for j in range(src_len):
            s_x   = Variable(xp.array([x_data[i][-j-1] for i in range(batch_size)], dtype=np.int32))
            s_i   = self.IE(s_x)
            h     = self.EF(s_i, is_train)
        self.h = h

        return self.HH(h)

    # Decode one word
    def __call__ (self, x_data, train_ref=None, is_train=True, *args, **kwargs):
        # Unpacking
        xp = self._xp
        f  = self._activation

        # Decoding
        y = self.WS(self.h)
        
        if train_ref is not None:
            wt = Variable(xp.array(train_ref.data, dtype=np.int32))
        else:
            wt = Variable(xp.array(UF.argmax(y.data), dtype=np.int32))
        w_n = self.OE(wt)
        
        # Updating
        self.h  = self.HH(self.EF(w_n, is_train))
        return DecodingOutput(y)
   
   def clean_state(self):
       pass

