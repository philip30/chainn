import sys
import numpy as np
import chainer.functions as F

# Chainer
from chainer import FunctionSet, Variable, optimizers, cuda

# Chainn
from chainn import functions as UF
from chainn.model import EncoderDecoder
from chainn.util.io import ModelFile
from chainn.util.output import DecodingOutput
from chainn.util import Vocabulary

# By Philip Arthur (philip.arthur30@gmail.com)
# This program is an implementation of NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
# (Bahdanau et al., 2015)
# http://arxiv.org/pdf/1409.0473v6.pdf

class Attentional(EncoderDecoder):
    # Architecture from: https://github.com/odashi/chainer_examples
    def _construct_model(self):
        I, O = self._input, self._output
        H, E = self._hidden, self._embed
        model = FunctionSet(
            # shared embedding
            w_E = F.EmbedID(I, E),
            # forward Encoder
            w_WF = F.Linear(E, 4 * H),
            w_UF = F.Linear(H, 4 * H),
            # backward Encoder
            w_WB = F.Linear(E, 4 * H),
            w_UB = F.Linear(H, 4 * H),
            # alignment model
            w_Wa = F.Linear(H, H),
            w_UaF = F.Linear(H, H),
            w_UaB = F.Linear(H, H),
            w_va = F.Linear(H, 1),
            # decoder
            w_Ws = F.Linear(H, H),
            w_V0 = F.EmbedID(O, 4 * H),
            w_U0 = F.Linear(H, 4 * H),
            w_C0F = F.Linear(H, 4 * H),
            w_C0B = F.Linear(H, 4 * H),
            w_ti  = F.Linear(H, O)
        )
        return model

    def _encode(self, src_batch):
        xp, hidden = self._xp, self._hidden
        m          = self._model
        row_len    = len(src_batch)
        col_len    = len(src_batch[0])
        
        # Encoding
        h    = []
        s_cf = Variable(xp.zeros((row_len, hidden), dtype=np.float32)) # cell state
        s_pf = Variable(xp.zeros((row_len, hidden), dtype=np.float32)) # outgoing signal
        s_cb = Variable(xp.zeros((row_len, hidden), dtype=np.float32)) # backward cell state
        s_pb = Variable(xp.zeros((row_len, hidden), dtype=np.float32)) # backward outgoing signal
        for j in range(col_len):
            # forward
            s_x        = Variable(xp.array([src_batch[i][j] for i in range(row_len)], dtype=np.int32))
            s_i        = F.tanh(m.w_E(s_x))
            s_cf, s_pf = F.lstm(s_cf, m.w_WF(s_i) + m.w_UF(s_pf))
            # backward
            s_xb       = Variable(xp.array([src_batch[i][-j-1] for i in range(row_len)], dtype=np.int32))
            s_ib       = F.tanh(m.w_E(s_xb))
            s_cb, s_pb = F.lstm(s_cb, m.w_WB(s_ib) + m.w_UB(s_pb))
            # concatenating them
            h.append((s_pf, s_pb))
        # Return the list of hidden state
        return h, src_batch

    def _decode(self, h, batch_size, generation_limit, update_callback, is_training):
        xp, hidden = self._xp, self._hidden
        m          = self._model
        row_len    = batch_size
        col_len    = generation_limit
        get_data   = lambda x: UF.to_cpu(self._use_gpu, x)
        TRG        = self._trg_voc
        save_alpha = True
        h, src     = h

        # Precompute U_a * h_j
        UaH = []
        for h_jf, h_jb in h:
            UaH.append(m.w_UaF(h_jf) + m.w_UaB(h_jb))

        # Decoding
        h1f, h1b = h[0]
        s        = F.tanh(m.w_Ws(h1b)) # initial state
        c        = Variable(xp.zeros((row_len, hidden), dtype=np.float32)) #initial lstm cell state value
        y        = Variable(xp.array([TRG["<s>"] for _ in range(row_len)], dtype=np.int32))
        output_l = [[] for _ in range(row_len)] # output list
        output_a = [[] for _ in range(row_len)] # alignment
        y_state  = {"y": y}                     # state containing last translated word
        for j in range(col_len):
            # Calculating e
            e     = []
            sum_e = 0
            for i in range(len(h)):
                e_ij   = F.exp(m.w_va(F.tanh(m.w_Wa(s) + UaH[i])))
                e.append(e_ij)
                sum_e += e_ij

            # Calculating alignment model
            c_f = Variable(xp.zeros((row_len, hidden), dtype=np.float32))
            c_b = Variable(xp.zeros((row_len, hidden), dtype=np.float32))
            alpha = [[] for _ in range(row_len)]
            for i in range(len(h)):
                alpha_ij = e[i] / sum_e
                h_f, h_b = h[i]
                c_f += F.reshape(F.batch_matmul(h_f, alpha_ij), (row_len, hidden)) # Forward
                c_b += F.reshape(F.batch_matmul(h_b, alpha_ij), (row_len, hidden)) # Backward
                
                if save_alpha:
                    for k in range(row_len):
                        alpha[k].append(get_data(alpha_ij.data)[k][0])

            # Generate next word
            c, s = F.lstm(c, m.w_U0(s) + m.w_V0(y_state["y"]) + m.w_C0F(c_f) + m.w_C0B(c_b))
            r_y  = self._dictionary_consideration(src, m.w_ti(s), alpha)

            out  = get_data(r_y.data).argmax(1)
            
            for i in range(len(out)):
                output_l[i].append(out[i])
                if save_alpha:
                    output_a[i].append(alpha[i])
            
            # Calculate entropy or if it is testing,
            # Break when every sentence has "</s>" at ending
            break_signal = update_callback(j, r_y, out, output_l, y_state)
            
            if break_signal:
                break
        
        return DecodingOutput(decode=output_l, alignment=output_a)
    
    def _save_parameter(self, fp):
        m = self._model
        fp.write_embed(m.w_E)
        fp.write_linear(m.w_WF)
        fp.write_linear(m.w_UF)
        fp.write_linear(m.w_WB)
        fp.write_linear(m.w_UB)
        fp.write_linear(m.w_Wa)
        fp.write_linear(m.w_UaF)
        fp.write_linear(m.w_UaB)
        fp.write_linear(m.w_va)
        fp.write_linear(m.w_Ws)
        fp.write_embed(m.w_V0)
        fp.write_linear(m.w_U0)
        fp.write_linear(m.w_C0F)
        fp.write_linear(m.w_C0B)
        fp.write_linear(m.w_ti)
        
    def _load_parameter(self, fp):
        m = self._model
        fp.read_embed(m.w_E)
        fp.read_linear(m.w_WF)
        fp.read_linear(m.w_UF)
        fp.read_linear(m.w_WB)
        fp.read_linear(m.w_UB)
        fp.read_linear(m.w_Wa)
        fp.read_linear(m.w_UaF)
        fp.read_linear(m.w_UaB)
        fp.read_linear(m.w_va)
        fp.read_linear(m.w_Ws)
        fp.read_embed(m.w_V0)
        fp.read_linear(m.w_U0)
        fp.read_linear(m.w_C0F)
        fp.read_linear(m.w_C0B)
        fp.read_linear(m.w_ti)
        if self._dict is not None:
            self._dict = self._load_dictionary(self._dict)
   
