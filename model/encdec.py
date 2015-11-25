import numpy as np
import chainer.functions as F
import util.functions as UF

from chainer import FunctionSet, Variable, optimizers, cuda
from util.io import ModelFile
from util.vocabulary import Vocabulary
from .nmt import NMT


class EncoderDecoder(NMT):
    def save(self, fp):
        self._src_voc.save(fp)
        self._trg_voc.save(fp)
        print(self._embed, file=fp)
        print(self._input, file=fp)
        print(self._output, file=fp)
        print(self._hidden, file=fp)
        fp = ModelFile(fp)
        if self._use_gpu: self._model = self._model.to_cpu()
        self._save_parameter(fp)
        if self._use_gpu: self._model = self._model.to_gpu()
   
    def load(self, fp):
        self._src_voc = Vocabulary.load(fp)
        self._trg_voc = Vocabulary.load(fp)
        self._embed   = int(next(fp))
        self._input   = int(next(fp))
        self._output  = int(next(fp))
        self._hidden  = int(next(fp))
        self._model   = self._init_model()
        fp = ModelFile(fp)
        if self._use_gpu: self._model = self._model.to_cpu()
        self._load_parameter(fp)
        if self._use_gpu: self._model = self._model.to_gpu()

    """ 
    Privates 
    """
    # Architecture from: https://github.com/odashi/chainer_examples
    def _construct_model(self):
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
        return model

    def _forward_training(self, src_batch, trg_batch):
        h = self._encode(src_batch)
        return self._decode_training(h, trg_batch)
         
    def _forward_testing(self, src_batch):
        h = self._encode(src_batch)
        return self._decode_testing(h, len(src_batch))

    def _encode(self, src_batch):
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

    def _decode_training(self, h, trg_batch):
        # Decoding (Producing target tokens & counting loss function)
        c, p       = h_
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

    def _decode_testing(self, h, batch_size):
        c, p = h
        xp   = self._xp
        GEN  = self._gen_lim
        m    = self._model
        output_l = [[] for i in range(batch_size)]
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

            for i in range(batch_size):
                output_l[i].append(output[i])
                # Whether we have finished translate this particular sentence
                if i not in all_done and output[i] == EOS:
                    all_done.add(i)
            
            # We have finished all the sentences in this batch
            if len(all_done) == batch_size:
                break
            
        return output_l
    
    def _save_parameter(self, fp):
        m = self._model
        fp.write_embed(m.w_xi)
        fp.write_linear(m.w_ip)
        fp.write_linear(m.w_pp)
        fp.write_linear(m.w_pq)
        fp.write_linear(m.w_qj)
        fp.write_linear(m.w_jy)
        fp.write_embed(m.w_yq)
        fp.write_linear(m.w_qq)

    def _load_parameter(self, fp):
        m = self._model
        fp.read_embed(m.w_xi)
        fp.read_linear(m.w_ip)
        fp.read_linear(m.w_pp)
        fp.read_linear(m.w_pq)
        fp.read_linear(m.w_qj)
        fp.read_linear(m.w_jy)
        fp.read_embed(m.w_yq)
        fp.read_linear(m.w_qq)
    
