import numpy as np
import chainer.links as L
import chainer.functions as F

# Chainer
from chainer import Variable, ChainList

# Chainn
from chainn import functions as UF
from chainn.model import ChainnBasicModel
from chainn.util import DecodingOutput
from chainn.link import StackLSTM

# By Philip Arthur (philip.arthur30@gmail.com)
# This program is an implementation of Sequence to Sequence Learning with Neural Networks
# (Sutskever et al., 2013)
# http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

class EncoderDecoder(ChainnBasicModel):
    name = "encdec"    
    
    def _construct_model(self, input, output, hidden, depth, embed):
        I, O, E, H = input, output, embed, hidden
        self.encoder = Encoder(I, E, H, depth, self._dropout)
        self.decoder = Decoder(O, E, H, depth, self._dropout)
        return [self.encoder, self.decoder]
    
    # Encoding all the source sentence
    def reset_state(self, x_data, is_train=False, *args, **kwargs):
        s = self.encoder(x_data, is_train=is_train)
        self.h = self.decoder.reset(s)
        return self.h
    
    # Decode one word
    def __call__ (self, x_data, is_train=False, eos_disc=0.0, *args, **kwargs):
        # Calculate the score of all target word (not yet softmax)
        y = self.decoder(self.h)
        
        # To adjust brevity score during decoding
        if not is_train and eos_disc != 0.0:
            y = self._adjust_brevity(yp, eos_disc)

        return DecodingOutput(y)

    # Update decoding state
    def update(self, wt, is_train=False):
        self.h = self.decoder.update(wt, is_train=is_train)

    # Adjusting brevity during decoding
    def _adjust_brevity(self, yp, eos_disc):
        v = self._xp.ones(len(self._trg_voc), dtype=np.float32)
        v[self._trg_voc.eos_id()] = 1-eos_disc
        v  = F.broadcast_to(Variable(v), yp.data.shape)
        return yp * v

    def clean_state(self):
        self.h = None

    def get_state(self):
        return (self.encoder.EF.get_state(), self.encoder.EB.get_state(), self.decoder.DF.get_state())

    def set_state(self, state):
        self.encoder.EF.set_state(state[0])
        self.encoder.EB.set_state(state[1])
        self.decoder.DF.set_state(state[2])

class Encoder(ChainList):
    def __init__(self, I, E, H, depth, dropout_ratio):
        self.IE = L.EmbedID(I, E)
        self.EF = StackLSTM(E, H, depth, dropout_ratio)
        self.EB = StackLSTM(E, H, depth, dropout_ratio)
        self.AE = L.Linear(2*H, H)
        self.H  = H
        super(Encoder, self).__init__(self.IE, self.EF, self.EB, self.AE)

    def __call__(self, src, is_train=False, xp=np):
        # Unpacking
        B  = len(src)      # Batch Size
        N  = len(src[0])   # length of source
        H  = self.H
        src_col = lambda x: Variable(self.xp.array([src[i][x] for i in range(B)], dtype=np.int32))
        embed   = lambda e, x: e(self.IE(x), is_train=is_train)
        
        # State Reset
        self.EF.reset_state()
        self.EB.reset_state()
        
        # Forward + backward encoding
        fe, be = None, None
        for j in range(N):
            fe = embed(self.EF, src_col(j))
            be = embed(self.EB, src_col(-j-1))

        # Joining encoding together
        return self.AE(F.concat((fe, be), axis=1))

class Decoder(ChainList):
    def __init__(self, O, E, H, depth, dropout_ratio):
        self.DF = StackLSTM(E, H, depth, dropout_ratio)
        self.WS = L.Linear(H, O)
        self.OE = L.EmbedID(O, E)
        self.HE = L.Linear(H, E)
        super(Decoder, self).__init__(self.DF, self.WS, self.OE, self.HE)

    def reset(self, s, is_train=False):
        self.DF.reset_state()
        return self.DF(self.HE(s), is_train=is_train)

    def __call__(self, h):
        return self.WS(F.tanh(h))
   
    def update(self, wt, is_train=False):
        return self.DF(self.OE(wt), is_train=is_train)

