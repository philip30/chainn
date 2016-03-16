import numpy as np
import sys, math
from collections import defaultdict

from chainn.util import functions as UF
import chainer.functions as F
import chainer.links as L

# Chainer
from chainer import Variable, cuda

# Chainn
from chainn import functions as UF
from chainn.link import LSTM, LinearInterpolation
from chainn.model.basic import ChainnBasicModel
from chainn.model.nmt import Attentional

# By Philip Arthur (philip.arthur30@gmail.com)

eps = 0.001
class DictAttentional(Attentional):
    name = "dictattn" 

    def __init__(self, src_voc, trg_voc, args, *other, **kwargs):
        super(DictAttentional, self).__init__(src_voc, trg_voc, args, *other, **kwargs)
        self._dict = self._load_dictionary(args.dict)

    def reset_state(self, src, trg, *args, **kwargs):
        SRC = self._src_voc
        TRG = self._trg_voc
        dct = self._dict
        xp  = self._xp
        vocab_size = self._output
        batch_size = len(src)
        src_len = len(src[0])

        prob_dict = np.zeros((src_len, batch_size, vocab_size), dtype=np.float32)
        for j in range(src_len):
            for i in range(batch_size):
                src_word = SRC.tok_rpr(src[i][j])
                if src_word in dct:
                    for trg_word, p in dct[src_word].items():
                        prob_dict[j][i][TRG[trg_word]] += p
        self.prob_dict = F.swapaxes(Variable(xp.array(prob_dict, dtype=np.float32)), 0, 1)

        return super(DictAttentional, self).reset_state(src, trg, *args, **kwargs) 

    def _load_dictionary(self, dict_dir):
        if type(dict_dir) is not str:
            return dict_dir
        dct = defaultdict(lambda:{})
        with open(dict_dir) as fp:
            for line in fp:
                line = line.strip().split()
                src, trg = line[1], line[0]
                if src in self._src_voc and trg in self._trg_voc:
                    prob = float(line[2])
                    dct[src][trg] = prob

        return dict(dct)

    def _additional_score(self, y, a, src):
        batch_size = len(y.data)
        vocab_size = self._output
        xp         = self._xp
        src_len    = len(self.prob_dict)
        # Calculating dict prob
        y_dict = F.reshape(F.batch_matmul(a, self.prob_dict, transa=True), (batch_size, vocab_size))
        
        # Using dict prob
        yp = y + F.log(eps + y_dict)
        return yp

    @staticmethod
    def _load_details(fp, args, xp, SRC, TRG):
        args.dict = defaultdict(lambda:{})
        fp.read_2leveldict(args.dict)
        args.dict = dict(args.dict)
             
    def _save_details(self, fp):
        super(DictAttentional, self)._save_details(fp)
        fp.write_2leveldict(self._dict)
    
