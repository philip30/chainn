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
from chainn.chainer_component.links import LinearInterpolation
from chainn.model import ChainnBasicModel
from chainn.model.nmt import Attentional

# By Philip Arthur (philip.arthur30@gmail.com)

eps = 0.001
class DictAttentional(Attentional):
    name = "dictattn" 

    def __init__(self, src_voc, trg_voc, spec, *other, **kwargs):
        self._caching = spec.dict_caching
        self._method  = spec.dict_method
        super(DictAttentional, self).__init__(src_voc, trg_voc, spec, *other, **kwargs)
        self._src_voc  = src_voc
        self._trg_voc  = trg_voc
        self._dict     = self._load_dictionary(spec.dict, src_voc, trg_voc)
        self._unk_src  = None
        
    def _construct_model(self, input, output, hidden, depth, embed):
        parent_list = super(DictAttentional, self)._construct_model(input, output, hidden, depth, embed)
        
        if self._method == "linear":
            self.LI = LinearInterpolation()
            parent_list.append(self.LI)
        
        return parent_list
 
    def reset_state(self, src, *args, is_train=False, **kwargs):
        SRC = self._src_voc
        TRG = self._trg_voc
        dct = self._dict
        xp  = self._xp
        vocab_size = self._output
        batch_size = len(src)
        src_len = len(src[0])
        volatile = "off" if is_train else "on"

        prob_dict = np.zeros((batch_size, src_len, vocab_size), dtype=np.float32)
       
        for i in range(batch_size):
            for j in range(src_len):
                src_word = src[i][j]
                if src_word in dct:
                    if self._caching:
                        prob_dict[i][j] = self.calculate_global_cache_dict(src_word)
                    else:
                        prob_dict[i][j] = self.calculate_local_cache_dict(src_word, dct)
                else:
                    prob_dict[i][j] = self.unk_src_dict(self._src_voc, self._output)
        self.prob_dict = Variable(xp.array(prob_dict), volatile=volatile)
        return super(DictAttentional, self).reset_state(src, *args, is_train=is_train, **kwargs) 

    def clean_state(self):
        self.prob_dict = None

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
        sum_prob = 0
        for trg_word, p in dct.items():
            ret_prob[trg_word] += p
            sum_prob += p
        ret_prob[self._src_voc.unk_id()] = 1.0 - sum_prob

        return ret_prob

    def unk_src_dict(self, src_dict, output_size):
        if self._unk_src is None:
            self._unk_src = np.zeros((self._output), dtype=np.float32)
            self._unk_src[src_dict.unk_id()] = 1.0
        return self._unk_src

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
                trg, src = line[0], line[1]
                if src in src_voc and trg in trg_voc:
                    prob = float(line[2])
                    dct[src_voc[src]][trg_voc[trg]] = prob
        self._dict_dir = dict_dir
        dct = dict(dct)
        return self._compile_dictionary(dct) if self._caching else dct

    def _additional_score(self, y, a, src):
        batch_size = len(y.data)
        vocab_size = self._output
        xp         = self._xp
        src_len    = len(self.prob_dict)
        # Calculating dict prob
        is_prob=False
        y_dict = F.reshape(F.batch_matmul(self.prob_dict, a, transa=True), (batch_size, vocab_size))
        # Using dict prob
        if self._method == "bias":
            yp = y + F.log(eps + y_dict)
        elif self._method == "linear":
            yp = self.LI(y_dict, F.softmax(y))
            is_prob = True
        else:
            raise ValueError("Unrecognized dictionary method:", self._method)
        return yp, is_prob

    def get_specification(self):
        ret = super(DictAttentional, self).get_specification()
        ret["dict"]          = self._dict_dir
        ret["dict_caching"]  = self._caching
        ret["dict_method"]   = self._method
        return ret
    
    def report(self, stream, verbosity=0):
        if verbosity >= 1:
            if self._method == "linear":
                x = self.LI.W.data
                print("Interpolation coefficient x=%f, sigmoid(x)=%f" % (x, sigmoid(x)), file=stream)

# Some supports
import math
def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))
