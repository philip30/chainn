import numpy as np
import chainer.functions as F

# Chainer
from chainer import Variable

# Chainn
from chainn import functions as UF
from chainn.model.basic import ChainnBasicModel
from chainn.util import DecodingOutput

# By Philip Arthur (philip.arthur30@gmail.com)
# This program is an implementation of Sequence to Sequence Learning with Neural Networks
# (Sutskever et al., 2013)
# http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

class EncoderDecoder(ChainnBasicModel):
    name = "encdec"    
    
    def _construct_model(self, input, output, hidden, depth, embed):
        assert(depth >= 1)
        I, O, E, H = input, output, embed, hidden
        ret = []
        # Encoder
        ret.append(F.EmbedID(I, E))      # w_xi 0
        ret.append(F.Linear(E, 4*H))     # w_ip 1
        ret.append(F.Linear(H, 4*H))     # w_pp 2
        # Decoder
        ret.append(F.Linear(H, 4*H))     # w_pq 3
        ret.append(F.Linear(H, E))       # w_qj 4
        ret.append(F.Linear(E, O))       # w_jy 5
        ret.append(F.EmbedID(O, 4*H))    # w_yq 6 
        ret.append(F.Linear(H, 4*H))     # w_qq 7
        return ret
    
    # Encoding all the source sentence
    def reset_state(self, x_data, y_data):
        # Unpacking
        xp, hidden  = self._xp, self._hidden
        row_len     = len(x_data)
        col_len     = len(x_data[0])
        f           = self._activation
        
        # Model Weight
        XI, IP, PP, PQ = self[0:4]

        # Encoding (Reading up source sentence)
        s_c = Variable(xp.zeros((row_len, hidden), dtype=np.float32)) # cell state
        s_p = Variable(xp.zeros((row_len, hidden), dtype=np.float32)) # outgoing signal
        for j in reversed(range(col_len)):
            s_x      = Variable(xp.array([x_data[i][j] for i in range(row_len)], dtype=np.int32))
            s_i      = f(XI(s_x))
            s_c, s_p = F.lstm(s_c, IP(s_i) + PP(s_p))
        self._h = F.lstm(s_c, PQ(s_p))
        return self._h

    # Decode one word
    def __call__ (self, x_data, train_ref=None, update=True):
        # Unpacking
        xp = self._xp
        f  = self._activation
        QJ, JY, YQ, QQ = self[4:8]
        
        # Decoding
        s_c, s_q = self._h
        s_j      = f(QJ(s_q))
        y        = JY(s_j)
        
        if update:
            if train_ref is not None:
                upd = Variable(xp.array(train_ref.data, dtype=np.int32))
            else:
                upd = Variable(xp.array(UF.argmax(y.data), dtype=np.int32))
    
            # Updating
            self._h  = F.lstm(s_c, YQ(upd) + QQ(s_q))
        return DecodingOutput(y)
    
