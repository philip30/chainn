import numpy as np
import chainer.functions as F
import chainer.links as L

# Chainer
from chainer import Variable

# Chainn
from chainn import functions as UF
from chainn.link import StackLSTM
from chainn.model.basic import ChainnBasicModel
from chainn.util import DecodingOutput

# By Philip Arthur (philip.arthur30@gmail.com)
# This program is an implementation of Effective Approaches to Attention-based Neural Machine Translation
# (Luong et al., 2015)
# http://arxiv.org/pdf/1508.04025v5.pdf

class Attentional(ChainnBasicModel):
    name = "attn" 
    
    def _construct_model(self, input, output, hidden, depth, embed):
        I, O, E, H = input, output, embed, hidden
        ret = []
        self.IE = L.EmbedID(I,E)
        self.EF = StackLSTM(E,H,depth)
        self.EB = StackLSTM(E,H,depth)
        self.AE = L.Linear(2*H, 2*H)
        self.AH = L.Linear(2*H, 2*H)
        self.AS = L.Linear(2*H, 2*H)
        self.WC = L.Linear(4*H, H)
        self.WS = L.Linear(H, O)
        self.OE = L.EmbedID(O, E)

        # Shared Embedding
        ret.append(self.IE)         # IE
        # Encoder           
        ret.append(self.EF)         # EF
        ret.append(self.EB)         # EB
        # Alignment Weight
        ret.append(self.AE)
        ret.append(self.AH)
        ret.append(self.AS)
        # Decoder
        ret.append(self.WC)         # WC
        ret.append(self.WS)         # WS
        ret.append(self.OE)         # OE
        return ret
    
    def reset_state(self, x_data, y_data):
        batch_size = len(x_data)
        src_len    = len(x_data[0])
        xp = self._xp
        f  = self._activation
        is_train = y_data is not None
        self.EF.reset_state()
        self.EB.reset_state()

        s = [[0,0] for _ in range(src_len)]
        for j in range(src_len):
            s_x       = Variable(xp.array([x_data[i][j] for i in range(batch_size)], dtype=np.int32))
            s_xb      = Variable(xp.array([x_data[i][-j-1] for i in range(batch_size)], dtype=np.int32))
            s_i, s_ib = self.IE(s_x), self.IE(s_xb)
            hf, hb    = f(self.EF(s_i, is_train)), f(self.EB(s_ib, is_train))
            # concatenating them
            s[j][0]   = hf
            s[-j-1][1]   = hb
        for i in range(len(s)):
            s[i] = F.concat((s[i][0], s[i][1]), axis=1)
        self.h = s[-1]
        self.s = s
        return s
     
    def __call__ (self, x_data, train_ref=None, update=True, debug=False):
        src_len = len(x_data[0])
        batch_size = len(x_data)
        hidden = self._hidden
        xp = self._xp
        f  = self._activation
        is_train = train_ref is not None
        # Calculate alignment weights
        a = []
        total_a = 0
        for i in range(src_len):
            score = self._score(self.h, self.AE(self.s[i]), batch_size)
            a.append(score)
            total_a += score
        
        for i in range(len(a)):
            a[i] = a[i] / total_a

        # Calculate context vector
        c = 0
        for i in range(src_len):
            c += F.reshape(F.batch_matmul(self.s[i], a[i]), (batch_size, 2*hidden))
        ht = self.WC(F.concat((self.h, f(c)), axis=1))
        yp = self.WS(ht)
        
        # Enhance y
        y = self._additional_score(yp, a, x_data)

        # Calculate next hidden hidden state
        if update:
            if train_ref is not None:
                # Training
                wt = train_ref
            else:
                # Testing
                wt = Variable(xp.array(UF.argmax(y.data), dtype=np.int32))
            w_n = self.OE(wt)
            self.h = F.concat((f(self.EF(w_n, is_train)), f(self.EB(w_n, is_train))), axis=1)
        return DecodingOutput(y, a)

    def _score(self, h, s, batch_size=1):
        f = self._activation
        return F.exp(f(F.reshape(F.batch_matmul(self.AH(h), s, transa=True), (batch_size, 1))))

    def _additional_score(self, y, a, x_data):
        return y

