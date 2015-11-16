
import numpy as np
import chainer.functions as F
import util.functions as UF

from chainer import FunctionSet, Variable, optimizers
from util.settings import DecoderSettings as DS

class EncoderDecoder:
    """ 
    Constructor 
    """
    def __init__(self, optimizer, gc):
        self._model = self.__init_model()
        self._optimizer = optimizer
        self._gc = gc

    @staticmethod
    def new(optimizer=optimizers.SGD(), gradient_clip=10):
        self = EncoderDecoder(optimizer, gradient_clip)
        return self

    """ 
    Publics 
    """ 
    def init_params(self):
        UF.init_model_parameters(self._model, -0.08, 0.08)
        if DS.use_gpu:
            self._model = self._model.to_gpu()
        self._optimizer.setup(self._model)
   
    def train(self, src_batch, trg_batch):
        return self.__forward(src_batch, trg_batch)
    
    def decay_lr(self, decay_factor):
        self._optimizer.lr /= decay_factor
    
    def update(self, loss):
        self._optimizer.zero_grads()
        loss.backward()
        loss.unchain_backward()
        self._optimizer.clip_grads(self._gc)
        self._optimizer.update()
    
    def decoder(self, src_batch):
        pass

    def save(self, fp):
        pass

    def load(self, fp):
        pass

    """ 
    Privates 
    """
    # Architecture from: https://github.com/odashi/chainer_examples
    def __init_model(self):
        I, O = DS.input, DS.output
        H, E = DS.hidden, DS.embed
        model = FunctionSet(
            # Encoder
            w_xi = F.EmbedID(I, E),
            w_ip = F.Linear(E, 4 * H),
            w_pp = F.Linear(H, 4 * H),
            # Decoder
            w_pq = F.Linear(H, 4 * H),
            w_qj = F.Linear(H, E),
            w_jy = F.Linear(E, O),
            w_yq = F.EmbedID(O, 4 * H),
            w_qq = F.Linear(H, 4* H)
        )
        return model

    def __forward(self, src_batch, trg_batch):
        xp, hidden = DS.xp, DS.hidden
        m          = self._model
        row_len    = len(src_batch)
        col_len    = len(src_batch[0])
        SRC, TRG   = DS.src_voc, DS.trg_voc
        
        # Encoding (Reading up source sentence)
        s_c = Variable(xp.zeros((row_len, hidden), dtype=np.float32)) # cell state
        s_p = Variable(xp.zeros((row_len, hidden), dtype=np.float32)) # outgoing signal
        for j in reversed(range(col_len)):
            s_x      = Variable(xp.array([src_batch[i][j] for i in range(row_len)], dtype=np.int32))
            s_i      = F.tanh(m.w_xi(s_x))
            s_c, s_p = F.lstm(s_c, m.w_ip(s_i) + m.w_pp(s_p))
        
        # Decoding (Producing target tokens)
        accum_loss = Variable(xp.zeros((), dtype=np.float32))
        col_len    = len(trg_batch[0])
        output_l   = [[] for i in range(row_len)]
        s_c, s_q   = F.lstm(s_c, m.w_pq(s_p))
        for j in range(col_len):
            s_j = F.tanh(m.w_qj(s_q))
            r_y = m.w_jy(s_j)
            s_t = Variable(xp.array([trg_batch[i][j] for i in range(row_len)], dtype=np.int32))
            accum_loss += F.softmax_cross_entropy(r_y, s_t)
            output      = UF.to_cpu(DS.use_gpu, r_y.data).argmax(1)
            s_c, s_q    = F.lstm(s_c, m.w_yq(s_t) + m.w_qq(s_q))
            
            # Collecting Output
            for i in range(row_len):
                output_l[i].append(output[i])
        return output_l, accum_loss

