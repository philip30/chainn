import numpy as np
import chainer.functions as F
import util.functions as UF

from chainer import FunctionSet, Variable, optimizers, cuda
from util.io import ModelFile
from util.vocabulary import Vocabulary

class EncoderDecoder:
    """ 
    Constructor 
    """
    def __init__(self, src_voc=None, trg_voc=None,\
            optimizer=optimizers.SGD(), gc=10, hidden=5, \
            embed=5, input=5, output=5, compile=True, use_gpu=False,
            gen_limit=50):
        self._optimizer = optimizer
        self._gc = gc
        self._hidden = hidden
        self._embed = embed
        self._input = input
        self._output = output
        self._src_voc = src_voc
        self._trg_voc = trg_voc
        self._use_gpu = use_gpu
        self._gen_lim = gen_limit
        self._xp = cuda.cupy if use_gpu else np
        if compile:
            self._model = self.__init_model()
        
    """ 
    Publics 
    """ 
    def init_params(self):
        if self._use_gpu: self._model.to_cpu()
        UF.init_model_parameters(self._model, -0.08, 0.08)
        if self._use_gpu: self._model.to_gpu()
  
    def setup_optimizer(self):
        self._optimizer.setup(self._model)

    def train(self, src_batch, trg_batch):
        return self.__forward_training(src_batch, trg_batch)
    
    def decay_lr(self, decay_factor):
        self._optimizer.lr /= decay_factor
    
    def update(self, loss):
        self._optimizer.zero_grads()
        loss.backward()
        loss.unchain_backward()
        self._optimizer.clip_grads(self._gc)
        self._optimizer.update()
    
    def decode(self, src_batch):
        return self.__forward_testing(src_batch)

    def save(self, fp):
        self._src_voc.save(fp)
        self._trg_voc.save(fp)
        print(self._embed, file=fp)
        print(self._input, file=fp)
        print(self._output, file=fp)
        print(self._hidden, file=fp)
        fp = ModelFile(fp)
        if self._use_gpu: self._model = self._model.to_cpu()
        fp.write_embed(self._model.w_xi)
        fp.write_linear(self._model.w_ip)
        fp.write_linear(self._model.w_pp)
        fp.write_linear(self._model.w_pq)
        fp.write_linear(self._model.w_qj)
        fp.write_linear(self._model.w_jy)
        fp.write_embed(self._model.w_yq)
        fp.write_linear(self._model.w_qq)
        if self._use_gpu: self._model = self._model.to_gpu()
   
    def load(self, fp):
        self._src_voc = Vocabulary.load(fp)
        self._trg_voc = Vocabulary.load(fp)
        self._embed   = int(next(fp))
        self._input   = int(next(fp))
        self._output  = int(next(fp))
        self._hidden  = int(next(fp))
        self._model   = self.__init_model()
        fp = ModelFile(fp)
        if self._use_gpu: self._model = self._model.to_cpu()
        fp.read_embed(self._model.w_xi)
        fp.read_linear(self._model.w_ip)
        fp.read_linear(self._model.w_pp)
        fp.read_linear(self._model.w_pq)
        fp.read_linear(self._model.w_qj)
        fp.read_linear(self._model.w_jy)
        fp.read_embed(self._model.w_yq)
        fp.read_linear(self._model.w_qq)
        if self._use_gpu: self._model = self._model.to_gpu()

    def get_vocabularies(self):
        return self._src_voc, self._trg_voc

    """ 
    Privates 
    """
    # Architecture from: https://github.com/odashi/chainer_examples
    def __init_model(self):
        I, O = self._input, self._output
        H, E = self._hidden, self._embed
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
        return model.to_gpu() if self._use_gpu else model


    def __forward_training(self, src_batch, trg_batch):
        s_c, s_p = self.__encode(src_batch)
        return self.__decode_training(s_c, s_p, trg_batch)
         
    def __forward_testing(self, src_batch):
        s_c, s_p = self.__encode(src_batch)
        return self.__decode_testing(s_c, s_p, len(src_batch))

    def __encode(self, src_batch):
        xp, hidden = self._xp, self._hidden
        m          = self._model
        row_len    = len(src_batch)
        col_len    = len(src_batch[0])

        # Encoding (Reading up source sentence)
        s_c = Variable(xp.zeros((row_len, hidden), dtype=np.float32)) # cell state
        s_p = Variable(xp.zeros((row_len, hidden), dtype=np.float32)) # outgoing signal
        for j in reversed(range(col_len)):
            s_x      = Variable(xp.array([src_batch[i][j] for i in range(row_len)], dtype=np.int32))
            s_i      = F.tanh(m.w_xi(s_x))
            s_c, s_p = F.lstm(s_c, m.w_ip(s_i) + m.w_pp(s_p))
        return s_c, s_p

    def __decode_training(self, c, p, trg_batch):
        # Decoding (Producing target tokens & counting loss function)
        xp, m      = self._xp, self._model
        row_len    = len(trg_batch)
        col_len    = len(trg_batch[0])
        output_l   = [[] for i in range(row_len)]
        accum_loss = 0
        s_c, s_q   = F.lstm(c, m.w_pq(p))
        for j in range(col_len):
            s_j = F.tanh(m.w_qj(s_q))
            r_y = m.w_jy(s_j)
            s_t = Variable(xp.array([trg_batch[i][j] for i in range(row_len)], dtype=np.int32))
            accum_loss += F.softmax_cross_entropy(r_y, s_t)
            output      = UF.to_cpu(self._use_gpu, r_y.data).argmax(1)
            s_c, s_q    = F.lstm(s_c, m.w_yq(s_t) + m.w_qq(s_q))
            
            # Collecting Output
            for i in range(row_len):
                output_l[i].append(output[i])
        return output_l, accum_loss

    def __decode_testing(self, c, p, trg_len):
        xp  = self._xp
        GEN = self._gen_lim
        m   = self._model
        output_l = [[] for i in range(trg_len)]
        EOS = self._trg_voc[self._trg_voc.get_eos()]
        all_done = set()
        # Decoding
        s_c, s_q = F.lstm(c, m.w_pq(p))
        for j in range(GEN):
            s_j    = F.tanh(m.w_qj(s_q))
            r_y    = m.w_jy(s_j)
            output = UF.to_cpu(self._use_gpu, r_y.data).argmax(1)
            outvar = Variable(xp.array(output, dtype=np.int32))
            s_c, s_q = F.lstm(s_c, m.w_yq(outvar) + m.w_qq(s_q))

            for i in range(trg_len):
                output_l[i].append(output[i])
                # Whether we have finished translate this particular sentence
                if i not in all_done and output[i] == EOS:
                    all_done.add(i)
            
            # We have finished all the sentences in this batch
            if len(all_done) == trg_len:
                break
            
        return output_l

