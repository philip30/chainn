import numpy as np
import sys, math
from collections import defaultdict

import chainer.functions as F
import chainer.links as L

# Chainer
from chainer import Variable, cuda

# Chainn
from chainn import functions as UF
from chainn.link import LSTM, LinearInterpolation
from chainn.model.basic import ChainnBasicModel
from chainn.model.nmt import EffectiveAttentional

# By Philip Arthur (philip.arthur30@gmail.com)

eps = 0.001
class DictAttentional(EffectiveAttentional):
    name = "dictattn" 

    def __init__(self, src_voc, trg_voc, args, *other, **kwargs):
        super(DictAttentional, self).__init__(src_voc, trg_voc, args, *other, **kwargs)
        self._dict = self._load_dictionary(args.dict)

    def _construct_model(self, input, output, hidden, depth, embed):
        ret = super(DictAttentional, self)._construct_model(input, output, hidden, depth, embed)
        self.WD = LinearInterpolation()
        ret.append(self.WD)
        return ret
 
    def _load_dictionary(self, dict_dir):
        if type(dict_dir) is not str:
            return dict_dir
        dct = defaultdict(lambda:{})
        mdc = defaultdict(lambda:0)
        with open(dict_dir) as fp:
            for line in fp:
                line = line.strip().split()
                src, trg = line[1], line[0]
                if src in self._src_voc and trg in self._trg_voc:
                    prob = float(line[2])
                    dct[src][trg] = prob
                    #if prob > mdc[src]:
                    #    mdc[src] = prob
                    #    dct[src] = {trg:prob}

        return dict(dct)

    def _additional_score(self, y, a, src):
        SRC = self._src_voc
        TRG = self._trg_voc
        dct = self._dict
        f   = self._activation
        # Copy value from a
        alpha = []
        for row in a:
            alpha_row = []
            for col in row.data:
                alpha_row.append(float(col))
            alpha.append(alpha_row)

        # Calculating dict prob
        y_dict = [[0 for _ in range(len(TRG))] for _ in range(len(src))]
        for i, batch in enumerate(src):
            for j, src_word in enumerate(batch):
                src_word = SRC.tok_rpr(src_word)
                if src_word in dct:
                    for trg_word, prob in dct[src_word].items():
                        y_dict[i][TRG[trg_word]] += prob * alpha[j][i]
        y_dict = F.log(eps + Variable(self._xp.array(y_dict, dtype=np.float32)))
    
        # Linear Interpolation
        y = self.WD(y_dict, y)
        return y

    @staticmethod
    def _load_details(fp, args, xp, SRC, TRG):
        args.dict = defaultdict(lambda:{})
        fp.read_2leveldict(args.dict)
        args.dict = dict(args.dict)
             
    def _save_details(self, fp):
        super(DictAttentional, self)._save_details(fp)
        fp.write_2leveldict(self._dict)
    
    def report(self):
        UF.trace("W:", str(self.WD.W.data))

