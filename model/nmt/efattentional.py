import numpy as np
import chainer.functions as F
import chainer.links as L

# Chainer
from chainer import Variable, ChainList

# Chainn
from chainn import functions as UF
from chainn.link import StackLSTM
from chainn.model.basic import ChainnBasicModel
from chainn.util import DecodingOutput

# By Philip Arthur (philip.arthur30@gmail.com)
# This program is an implementation of Effective Approaches to Attention-based Neural Machine Translation
# (Luong et al., 2015)
# http://arxiv.org/pdf/1508.04025v5.pdf

class AlignmentModel(ChainList):
    def __init__(self, H, depth):
        l = []
        for i in range(depth):
            l.append(L.Linear(H, H, nobias=True, initialW=np.random.uniform(-0.1, 0.1, (H,H))))
        l.append(L.Linear(H,1))
        super(AlignmentModel, self).__init__(*l)

    def __call__(self, inp):
        ret = None
        for layer in self:
            ret = F.tanh(layer(inp if ret is None else ret))
        return ret


class Attentional(ChainnBasicModel):
    name = "attn" 
    
    def _construct_model(self, input, output, hidden, depth, embed):
        I, O, E, H = input, output, embed, hidden
        ret = []
        self.IE = L.EmbedID(I,E)
        self.EF = StackLSTM(E,H,depth)
        self.EB = StackLSTM(E,H,depth)
        self.AE = L.Linear(H, H)
        self.HH = L.Linear(H, H)
        self.align = AlignmentModel(2*H, 2)
        self.WC = L.Linear(2*H, H)
        self.WS = L.Linear(H, O)
        self.OE = L.EmbedID(O, E)

        # Shared Embedding
        ret.append(self.IE)         # IE
        # Encoder           
        ret.append(self.EF)         # EF
        ret.append(self.EB)         # EB
        ret.append(self.AE)
        # Alignment Model
        ret.append(self.align)
        ret.append(self.HH)
        # Decoder
        ret.append(self.WC)         # WC
        ret.append(self.WS)         # WS
        ret.append(self.OE)         # OE
        return ret
    
    def reset_state(self, x_data, y_data, is_train=True):
        batch_size = len(x_data)
        src_len    = len(x_data[0])
        hidden_size = self._hidden
        xp = self._xp
        f  = self._activation
        self.EF.reset_state()
        self.EB.reset_state()
        # Forward + backward encoding
        s = [[0,0] for _ in range(src_len)]
        for j in range(src_len):
            s_x       = Variable(xp.array([x_data[i][j] for i in range(batch_size)], dtype=np.int32))
            s_xb      = Variable(xp.array([x_data[i][-j-1] for i in range(batch_size)], dtype=np.int32))
            hf, hb    = self.EF(self.IE(s_x), is_train), self.EB(self.IE(s_xb), is_train)
            # concatenating them
            s[j][0]    = hf
            s[-j-1][1] = hb

        # Joining the encoding data together
        S = None
        for i in range(len(s)):
            s_i = self.AE(s[i][0]) + s[i][1]
            if i == len(s)-1:
                self.h = s_i
            s_i = F.reshape(s_i, (batch_size, hidden_size, 1))
            S = s_i if S is None else F.concat((S, s_i), axis=2)

        self.s = S
        return S
     
    def __call__ (self, x_data, train_ref=None, update=True, is_train=True, debug=False):
        xp = self._xp
        src_len = len(x_data[0])
        batch_size = len(x_data)
        hidden_size = self._hidden
        f  = self._activation

        # Calculate alignment weights
        s, h = self.s, self.h
        ### Making them into same shapes
        h = F.reshape(f(self.HH(h)), (batch_size, hidden_size, 1))
        h = F.reshape(F.swapaxes(F.concat(F.broadcast(h, s), axis=1), 1, 2), (batch_size * src_len, 2 * hidden_size))
        ### Aligning them
        a = F.exp(f(self.align(h)))
        ### Restore the shape
        a = F.reshape(a, (batch_size, src_len))
        ### Renormalizing with the norm
        Z = F.reshape(1/F.sum(a, axis=1), (batch_size, 1))
        a = F.batch_matmul(a, Z)
        ### This is the old way
 #       a = F.exp(f(F.reshape(F.batch_matmul(h, s, transa=True), (batch_size, src_len, 1))))
 #       a = F.reshape(F.batch_matmul(a, 1/F.sum(a, axis=1)), (batch_size, src_len))

        # Calculate context vector
        c = F.reshape(F.batch_matmul(s, a), (batch_size, hidden_size))
        ht = self.WC(F.concat((self.h, c), axis=1))
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
            w_nf = self.EF(w_n, is_train)
            w_nb = self.EB(w_n, is_train)
            self.h = self.AE(w_nf) + w_nb
        return DecodingOutput(y, a)

    def _additional_score(self, y, a, x_data):
        return y

