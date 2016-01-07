import numpy as np
import chainer.functions as F
import chainer.links as L

# Chainer
from chainer import Variable

# Chainn
from chainn import functions as UF
from chainn.link import LSTM
from . import EncoderDecoder

# By Philip Arthur (philip.arthur30@gmail.com)
# This program is an implementation of Effective Approaches to Attention-based Neural Machine Translation
# (Luong et al., 2015)
# http://arxiv.org/pdf/1508.04025v5.pdf

class EffectiveAttentional(EncoderDecoder):
    name = "efattn" 

    def __init__(self, *args, **kwargs):
        super(EffectiveAttentional, self).__init__(*args, **kwargs)

    # Architecture from: https://github.com/odashi/chainer_examples
    def _construct_model(self, input, output, hidden, depth, embed):
        I, O, E, H = input, output, embed, hidden
        ret = []
        # Shared Embedding
        ret.append(L.EmbedID(I,E))          # IE
        # Encoder           
        ret.append(LSTM(E, H))              # EF
        ret.append(LSTM(E, H))              # EB
        # Decoder
        ret.append(LSTM(E, H))              # DF
        ret.append(LSTM(E, H))              # DB
        # Alignment Weight
        ret.append(L.Linear(4*H, H))        # WC
        ret.append(L.Linear(H, O))          # WS
        ret.append(L.EmbedID(O, E))         # OE
        return ret
    
    def reset_state(self, x_data):
        batch_size = len(x_data)
        src_len    = len(x_data[0])
        xp = self._xp
        f  = self._activation
        IE, EF, EB, DF, DB = self[0:5]
        EF.reset_state()
        EB.reset_state()
        DF.reset_state()
        DB.reset_state()

        s = [[0,0] for _ in range(src_len)]
        for j in range(src_len):
            s_x       = Variable(xp.array([x_data[i][j] for i in range(batch_size)], dtype=np.int32))
            s_xb      = Variable(xp.array([x_data[i][-j-1] for i in range(batch_size)], dtype=np.int32))
            s_i, s_ib = f(IE(s_x)), f(IE(s_xb))
            hf, hb    = EF(s_i), EB(s_ib)
            # concatenating them
            s[j][0]   = hf
            s[-j-1][1]   = hb
        for i in range(len(s)):
            s[i] = F.concat((s[i][0], s[i][1]), axis=1)
        self.h = s[-1]
        self.s = s
        DF.c = EF.c
        DB.c = EB.c
        return s
     
    def __call__ (self, x_data, train_ref=None, update=True):
        DF, DB = self[3], self[4]
        WC, WS = self[5], self[6]
        OE     = self[7]
        src_len = len(x_data[0])
        batch_size = len(x_data)
        hidden = self._hidden
        xp = self._xp
        # Calculate alignment weights
        a = []
        total_a = 0
        for i in range(src_len):
            score = self._score(self.h, self.s[i], batch_size)
            a.append(score)
            total_a += score
        
        for i in range(len(a)):
            a[i] = a[i] / total_a
        
        # Calculate context vector
        c = 0
        for i in range(src_len):
            c += F.reshape(F.batch_matmul(self.s[i], a[i]), (batch_size, 2*hidden))
        ht = WC(F.concat((self.h, c), axis=1))
        y = WS(ht)
        
        # Calculate next hidden hidden state
        if update:
            if train_ref is not None:
                # Training
                wt = Variable(xp.array(train_ref.data, dtype=np.int32))
            else:
                # Testing
                wt = Variable(xp.array(UF.argmax(y.data), dtype=np.int32))
            w_n = OE(wt)
            self.h = F.concat((DF(w_n), DB(w_n)), axis=1)
        return y

    def _score(self, h, s, batch_size=1):
        return F.exp(F.reshape(F.batch_matmul(h, s, transa=True), (batch_size, 1)))

