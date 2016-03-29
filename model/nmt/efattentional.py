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

DROPOUT_RATIO = 0.5
class Attentional(ChainnBasicModel):
    name = "attn" 
    
    def _construct_model(self, input, output, hidden, depth, embed):
        I, O, E, H = input, output, embed, hidden
        self.encoder   = Encoder(I, E, H, depth)
        self.attention = AttentionLayer()
        self.decoder   = Decoder(O, E, H, depth)
        return [self.encoder, self.attention, self.decoder]
    
    # Encode all the words in the input sentence
    def reset_state(self, x_data, y_data, is_train=False, *args, **kwargs):
        self.s, s_n = self.encoder(x_data, is_train=is_train, xp=self._xp)
        self.h = self.decoder.reset(s_n, is_train=is_train)
        return self.s
    
    # Produce one target word
    def __call__ (self, x_data, train_ref=None, is_train=False, eos_disc=0.0, *args, **kwargs):
        # Calculate alignment weights between hidden state and source vector context
        a  = self.attention(self.h, self.s)
        
        # Calculate the score of all target word (not yet softmax)
        yp = self.decoder(self.s, a, self.h)
        
        # To adjust brevity score during decoding
        if train_ref is None and eos_disc != 0.0:
            yp = self._adjust_brevity(yp, eos_disc)

        # Enhance y
        y = self._additional_score(yp, a, x_data)
        
        # Conceive the next state
        self.h = self._decode_next(y, train_ref=train_ref, is_train=is_train)
        return DecodingOutput(y, a)

    # Adjusting brevity during decoding
    def _adjust_brevity(self, yp, eos_disc):
        v = self._xp.ones(len(self._trg_voc), dtype=np.float32)
        v[self._trg_voc.eos_id()] = 1-eos_disc
        v  = F.broadcast_to(Variable(v), yp.data.shape)
        return yp * v

    # Update the RNN state 
    def _decode_next(self, y, train_ref, is_train=False):
        if train_ref is not None and is_train:
            # Training
            wt = train_ref
        else:
            # Testing
            wt = Variable(self._xp.array(UF.argmax(y.data), dtype=np.int32))
        return self.decoder.update(wt, is_train=is_train)

    # Whether we want to change y score by linguistic resources?
    def _additional_score(self, y, a, x_data):
        return y

    def clean_state(self):
        self.h = None
        self.s = None

class Encoder(ChainList):
    def __init__(self, I, E, H, depth):
        self.IE = L.EmbedID(I, E)
        self.EF = StackLSTM(E, H, depth, DROPOUT_RATIO)
        self.EB = StackLSTM(E, H, depth, DROPOUT_RATIO)
        self.AE = L.Linear(2*H, H)
        self.H  = H
        super(Encoder, self).__init__(self.IE, self.EF, self.EB, self.AE)

    def __call__(self, src, is_train=False, xp=np):
        # Some namings
        B  = len(src)      # Batch Size
        N  = len(src[0])   # length of source
        H  = self.H
        src_col = lambda x: Variable(self.xp.array([src[i][x] for i in range(B)], dtype=np.int32))
        embed   = lambda e, x: e(self.IE(x), is_train=is_train)
        bi_rnn  = lambda x, y: self.AE(F.concat((x[0], y[1]), axis=1))
        concat_source = lambda S, s: s if S is None else F.concat((S, s), axis=2)
        # State Reset
        self.EF.reset_state()
        self.EB.reset_state()
       
        # Forward + backward encoding
        s = []
        for j in range(N):
            s.append((
                embed(self.EF, src_col(j)),
                embed(self.EB, src_col(-j-1))
            ))
        
        # Joining the encoding data together
        S = None
        for j in range(N):
            s_j = bi_rnn(s[j], s[-j-1])
            S = concat_source(S, F.reshape(s_j, (B, H, 1)))
        S = F.swapaxes(S, 1, 2)
        return S, s_j

class AttentionLayer(ChainList):
    def __init__(self):
        super(AttentionLayer, self).__init__()
    
    def __call__(self, h, s):
        return self._dot(h, s)

    def _dot(self, h, s):
        return F.softmax(F.batch_matmul(s, h))

class Decoder(ChainList):
    def __init__(self, O, E, H, depth):
        self.DF = StackLSTM(E, H, depth, DROPOUT_RATIO)
        self.WS = L.Linear(H, O)
        self.WC = L.Linear(2*H, H)
        self.OE = L.EmbedID(O, E)
        self.HE = L.Linear(H, E)
        super(Decoder, self).__init__(self.DF, self.WS, self.WC, self.OE, self.HE)
    
    def __call__(self, s, a, h):
        B = len(s.data)
        H = len(h.data[0])
        c = F.reshape(F.batch_matmul(a, s, transa=True), (B, H))
        ht = F.tanh(self.WC(F.concat((h, c), axis=1)))
        return self.WS(ht)

    # Conceive the first state of decoder based on the last state of encoder
    def reset(self, s, is_train=False):
        self.DF.reset_state()
        return self.DF(self.HE(s), is_train=is_train)

    def update(self, wt, is_train=False):
        return self.DF(self.OE(wt), is_train=is_train)

