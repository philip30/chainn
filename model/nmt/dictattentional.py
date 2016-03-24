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
        self._caching = args.dict_caching if hasattr(args, "dict_caching") else False
        self._dict    = self._load_dictionary(args.dict, src_voc, trg_voc)
        
    def reset_state(self, src, trg, *args, **kwargs):
        SRC = self._src_voc
        TRG = self._trg_voc
        dct = self._dict
        xp  = self._xp
        vocab_size = self._output
        batch_size = len(src)
        src_len = len(src[0])

        prob_dict = np.zeros((batch_size, src_len, vocab_size), dtype=np.float32)
       
        for i in range(batch_size):
            for j in range(src_len):
                src_word = src[i][j]
                if src_word in dct:
                    if self._caching:
                        prob_dict[i][j] = self.calculate_global_cache_dict(src_word)
                    else:
                        prob_dict[i][j] = self.calculate_local_cache_dict(src_word, dct)
                    
        self.prob_dict = Variable(xp.array(prob_dict, dtype=np.float32))
        return super(DictAttentional, self).reset_state(src, trg, *args, **kwargs) 

    def calculate_global_cache_dict(self, src_word):
        return self._dict[src_word]
        
    def calculate_local_cache_dict(self, src_word, dct, cache={}):
        if src_word in cache:
            dict_vector = cache[src_word]
        else:
            dict_vector = self.calculate_dict_vector(dct[src_word])
            cache[src_word] = dict_vector
        return dict_vector

    def calculate_dict_vector(self, dct):
        ret_prob = np.zeros((self._output), dtype=np.float32)
        for trg_word, p in dct.items():
            ret_prob[trg_word] += p
        return ret_prob

    def _compile_dictionary(self, dct):
        ret = {}
        for src in dct:
            ret[src] = self.calculate_dict_vector(dct[src])
        return ret

    def _load_dictionary(self, dict_dir, src_voc, trg_voc):
        if type(dict_dir) is not str:
            return dict_dir
        dct = defaultdict(lambda:{})
        with open(dict_dir) as fp:
            for line in fp:
                line = line.strip().split()
                src, trg = line[1], line[0]
                if src in self._src_voc and trg in self._trg_voc:
                    prob = float(line[2])
                    dct[self._src_voc[src]][self._trg_voc[trg]] = prob
        self._dict_dir = dict_dir
        dct = dict(dct)
        return self._compile_dictionary(dct) if self._caching else dct

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
        args.dict = fp.read()
        args.dict_caching = fp.read() == "True"

    def _save_details(self, fp):
        fp.write(self._dict_dir)
        fp.write(str(self._caching))
    
