import numpy as np
import chainer.functions as F
import chainer.links as L

# Chainer
from chainer import Variable, ChainList

# Chainn
from chainn import functions as UF
from chainn.chainer_component.links import StackLSTM
from chainn.model.nmt import EncoderDecoder
from chainn.util import DecodingOutput

# By Philip Arthur (philip.arthur30@gmail.com)
# This program is an implementation of Effective Approaches to Attention-based Neural Machine Translation
# (Luong et al., 2015)
# http://arxiv.org/pdf/1508.04025v5.pdf

class Attentional(EncoderDecoder):
    name = "attn" 
    
    def __init__(self, src_voc, trg_voc, args, *other, **kwargs):
        self._attention_type = args.attention_type if hasattr(args, "attention_type") else "dot"
        super(Attentional, self).__init__(src_voc, trg_voc, args, *other, **kwargs)

    def _construct_model(self, input, output, hidden, depth, embed):
        I, O, E, H = input, output, embed, hidden
        self.encoder   = Encoder(I, E, H, depth, self._dropout)
        self.attention = AttentionLayer(H, self._attention_type)
        self.decoder   = Decoder(O, E, H, depth, self._dropout)
        return [self.encoder, self.attention, self.decoder]
    
    # Encode all the words in the input sentence
    def reset_state(self, x_data, is_train=False, *args, **kwargs):
        self.s, s_n = self.encoder(x_data, is_train=is_train, xp=self._xp)
        self.h = self.decoder.reset(s_n, is_train=is_train)
        return self.s
    
    # Produce one target word
    def __call__ (self, x_data, is_train=False, eos_disc=0.0, *args, **kwargs):
        # Calculate alignment weights between hidden state and source vector context
        a  = self.attention(self.h, self.s)
        
        # Calculate the score of all target word (not yet softmax)
        yp = self.decoder(self.s, a, self.h)
        
        # Enhance y
        y, is_prob = self._additional_score(yp, a, x_data)
        if not is_prob:
            y = F.softmax(y)
        
        # To adjust brevity score during decoding
        if eos_disc != 0.0:
            y = self._adjust_brevity(y, eos_disc)

        return DecodingOutput(y, a)

    # Whether we want to change y score by linguistic resources?
    def _additional_score(self, y, a, x_data):
        return y, False

    def clean_state(self):
        self.h = None
        self.s = None
    
    def get_specification(self):
        ret = super(Attentional, self).get_specification()
        ret["attention_type"] = self._attention_type
        return ret

class Encoder(ChainList):
    def __init__(self, I, E, H, depth, dropout_ratio):
        self.IE = L.EmbedID(I, E)
        self.EF = StackLSTM(E, H, depth, dropout_ratio)
        self.EB = StackLSTM(E, H, depth, dropout_ratio)
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
    def __init__(self, hidden, attn_type):
        self._type = attn_type
        param = []
        if attn_type == "general":
            self.WA = L.Linear(hidden , hidden)
            param.append(self.WA)
        elif attn_type == "concat":
            self.WA = L.Linear(2 * hidden , 1)
            param.append(self.WA)
        super(AttentionLayer, self).__init__(*param)
        
    def __call__(self, h, s):
        if self._type == "dot":
            return self._dot(h, s)
        elif self._type == "general":
            return self._general(h, s)
        elif self._type == "concat":
            return self._concat(h, s)
        else:
            raise Exception("Unrecognized type:", self._type) 
    
    def _general(self, h, s):
        batch, src_len, hidden = s.data.shape
        param_s = F.reshape(self.WA(F.reshape(s, (batch * src_len, hidden))), (batch, src_len, hidden))
        return self._dot(h, param_s)

    def _concat(self, h, s):
        batch, src_len, hidden = s.data.shape
        concat_h  = F.reshape(F.concat(F.broadcast(F.expand_dims(h, 1), s), axis=1), (batch * src_len, 2* hidden))
        return F.softmax(F.reshape(self.WA(concat_h), (batch, src_len)))
                
    def _dot(self, h, s):
        return F.softmax(F.batch_matmul(s, h))

class Decoder(ChainList):
    def __init__(self, O, E, H, depth, dropout_ratio):
        self.DF = StackLSTM(E, H, depth, dropout_ratio)
        self.WS = L.Linear(H, O)
        self.WC = L.Linear(2*H, H)
        self.OE = L.EmbedID(O, E)
        self.HE = L.Linear(H, E)
        super(Decoder, self).__init__(self.DF, self.WS, self.WC, self.OE, self.HE)
    
    def __call__(self, s, a, h):
        c = F.reshape(F.batch_matmul(a, s, transa=True), h.data.shape)
        ht = F.tanh(self.WC(F.concat((h, c), axis=1)))
        return self.WS(ht)

    # Conceive the first state of decoder based on the last state of encoder
    def reset(self, s, is_train=False):
        self.DF.reset_state()
        return self.DF(self.HE(s), is_train=is_train)

    def update(self, wt, is_train=False):
        return self.DF(self.OE(wt), is_train=is_train)

