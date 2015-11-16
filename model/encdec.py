
import numpy as np
import chainer.functions as F
import util.functions as UF

from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
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
        #loss.unchain_backward()
        print("GC:",self._gc)
        self._optimizer.clip_grads(self._gc)
        self._optimizer.update()

    """ 
    Privates 
    """
    def __init_model(self):
        I, O = DS.input, DS.output
        H, E = DS.hidden, DS.embed
        model = FunctionSet(
            # Encoder
            embed = F.EmbedID(I, E),
            d_eh = F.Linear(E, 4 * H),
            d_hh = F.Linear(H, 4 * H),
            # Decoder
            e_hx = F.Linear(H, 4 * H),
            e_xe = F.Linear(H, E),
            e_ey = F.Linear(E, O),
            e_yh = F.EmbedID(O, 4 * H),
            e_hh = F.Linear(H, 4* H)
        )
        return model

    def __forward(self, src_batch, trg_batch):
        xp, hidden = DS.xp, DS.hidden
        m          = self._model
        row_len    = len(src_batch)
        col_len    = len(src_batch[0])
        
        # Encoding (Reading up source sentence)
        c = Variable(xp.zeros((row_len, hidden), dtype=np.float32)) # cell state
        h = Variable(xp.zeros((row_len, hidden), dtype=np.float32)) # outgoing signal
        for j in reversed(range(col_len)):
            x    = Variable(xp.array([src_batch[i][j] for i in range(row_len)], dtype=np.int32))
            e    = F.tanh(m.embed(x))
            c, h = F.lstm(c, m.d_eh(e) + m.d_hh(h))
        
        # Decoding (Producing target tokens)
        accum_loss = Variable(xp.zeros((), dtype=np.float32))
        col_len    = len(trg_batch[0])
        output_l   = [[] for i in range(row_len)]
        c, h       = F.lstm(c, m.e_hx(h))
        for j in range(col_len):
            e = F.tanh(m.e_xe(h))
            y = m.e_ey(e)
            print(y.data)
            t = Variable(xp.array([trg_batch[i][j] for i in range(row_len)], dtype=np.int32))
            print(t.data)
            accum_loss += F.softmax_cross_entropy(y, t)
            print(accum_loss.data)
            output      = UF.to_cpu(DS.use_gpu, y.data).argmax(1)
            c, h        = F.lstm(c, m.e_yh(t) + m.e_hh(h))
            
            # Collecting Output
            for i in range(row_len):
                output_l[i].append(output[i])
        return output_l, accum_loss

