import numpy as np
import chainer.functions as F
import chainer.links as L

# Chainer
from chainer import Variable

# Chainn
from chainn import functions as UF
from chainn.model.basic import ChainnBasicModel
from chainn.util import DecodingOutput

# By Philip Arthur (philip.arthur30@gmail.com)
# This program is an implementation of NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
# (Bahdanau et al., 2015)
# http://arxiv.org/pdf/1409.0473v6.pdf

class Attentional(ChainnBasicModel):
    name = "attn" 

    # Architecture from: https://github.com/odashi/chainer_examples
    def _construct_model(self, input, output, hidden, depth, embed):
        I, O, E, H = input, output, embed, hidden
        ret = []
        # Shared Embedding
        ret.append(L.EmbedID(I,E))          # w_E 0
        # Forward Encoder           
        ret.append(L.Linear(E, 4*H))        # w_WF 1
        ret.append(L.Linear(H, 4*H))        # w_UF 2
        # Backward Encoder
        ret.append(L.Linear(E, 4*H))        # w_WB 3
        ret.append(L.Linear(H, 4*H))        # w_UB 4
        # Alignment model
        ret.append(L.Linear(H, H))          # w_UaF 5
        ret.append(L.Linear(H, H))          # w_UaB 6
        ret.append(L.Linear(H, H))          # w_wa 7
        ret.append(L.Linear(H, H))          # w_Ws 8
        ret.append(L.Linear(H, 1))          # w_va 9
        # Decoding
        ret.append(L.EmbedID(O, 4*H))       # w_v0 10
        ret.append(L.Linear(H, 4*H))        # w_u0 11
        ret.append(L.Linear(H, 4*H))        # w_c0f 12
        ret.append(L.Linear(H, 4*H))        # w_cob 13
        ret.append(L.Linear(H, O))          # w_ti 14
        return ret

    # Reset the state of decoder by encoding source sent
    def reset_state(self, x_data, y_data):
        xp, hidden   = self._xp, self._hidden
        IE, EHF, HHF = self[0:3]
        EHB, HHB     = self[3:5]
        f            = self._activation
        batch_size   = len(x_data)
        src_len      = len(x_data[0])
        TRG          = self._trg_voc

        # Encoding
        h    = [[0,0] for _ in range(src_len)]
        s_cf = Variable(xp.zeros((batch_size, hidden), dtype=np.float32)) # cell state
        s_pf = Variable(xp.zeros((batch_size, hidden), dtype=np.float32)) # outgoing signal
        s_cb = Variable(xp.zeros((batch_size, hidden), dtype=np.float32)) # backward cell state
        s_pb = Variable(xp.zeros((batch_size, hidden), dtype=np.float32)) # backward outgoing signal
        for j in range(src_len):
            # forward
            s_x        = Variable(xp.array([x_data[i][j] for i in range(batch_size)], dtype=np.int32))
            s_i        = f(IE(s_x))
            s_cf, s_pf = F.lstm(s_cf, EHF(s_i) + HHF(s_pf))
            # backward
            s_xb       = Variable(xp.array([x_data[i][-j-1] for i in range(batch_size)], dtype=np.int32))
            s_ib       = f(IE(s_xb))
            s_cb, s_pb = F.lstm(s_cb, EHB(s_ib) + HHB(s_pb))
            # concatenating them
            h[j][0] = s_pf
            h[-j-1][1] = s_pb
   
        # Initial state for decoder
        h1f, h1b = h[0]
        UaF, UaB = self[5:7]
        WS       = self[7]
        s        = f(WS(h1b)) # initial state
        c        = Variable(xp.zeros((batch_size, hidden), dtype=np.float32)) #initial lstm cell state value

        # Precompute U_a * h_j
        UaH = []
        for h_jf, h_jb in h:
            UaH.append(UaF(h_jf) + UaB(h_jb))
        y_p = None

        # Encoding finished
        self.h = c, s, UaH, h, y_p
        return self.h

    def __call__ (self, x_data, train_ref=None, update=True):
        f, xp      = self._activation, self._xp
        batch_size = len(x_data)
        hidden     = self._hidden
        c, s, UaH, h, y_p = self.h
        WA, VA = self[8:10]
        V0, U0, C0F, C0B, TI = self[10:15]

        # Calculating e
        e     = []
        sum_e = 0
        for i in range(len(h)):
            e_ij = F.exp(VA(f(WA(s) + UaH[i])))
            e.append(e_ij)
            sum_e += e_ij
        
        # Calculating alignment model
        c_f = Variable(xp.zeros((batch_size, hidden), dtype=np.float32))
        c_b = Variable(xp.zeros((batch_size, hidden), dtype=np.float32))
        alpha = [[] for _ in range(batch_size)]
        for i in range(len(h)):
            alpha_ij = e[i] / sum_e
            h_f, h_b = h[i]
            c_f += F.reshape(F.batch_matmul(h_f, alpha_ij), (batch_size, hidden)) # Forward
            c_b += F.reshape(F.batch_matmul(h_b, alpha_ij), (batch_size, hidden)) # Backward
        
        # Calculate output layer
        w_p  = V0(y_p) if y_p is not None else 0
        c, s = F.lstm(c, U0(s) + C0F(c_f) + C0B(c_b) + w_p)
        y    = TI(s)

        if update:
            if train_ref is not None:
                y_p = Variable(xp.array(train_ref.data, dtype=np.int32))
            else:
                y_p = Variable(xp.array(UF.argmax(y.data), dtype=np.int32))
            self.h = c, s, UaH, h, y_p
        return DecodingOutput(y)

