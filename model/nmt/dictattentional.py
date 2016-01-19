import numpy as np
import sys
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

class DictAttentional(EffectiveAttentional):
    name = "dictattn" 

    def __init__(self, src_voc, trg_voc, args, *other, **kwargs):
        super(DictAttentional, self).__init__(src_voc, trg_voc, args, *other, **kwargs)
        self._dict = self._load_dictionary(args.dict)

    def _construct_model(self, *args, **kwargs):
        ret = super(DictAttentional, self)._construct_model(*args, **kwargs)
        self.WD = LinearInterpolation(args[1])
        ret.append(self.WD)
        return ret
 
    def _load_dictionary(self, dict_dir):
        if type(dict_dir) is not str:
            return dict_dir
        dct = defaultdict(lambda:{})
        with open(dict_dir) as fp:
            for line in fp:
                line = line.strip().split()
                dct[line[0]][line[1]] = float(line[2])
        return dict(dct)

    def _additional_score(self, y, a, src):
        TRG = self._trg_voc
        dct = self._dict
      
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
                if src_word in dct:
                    for trg_word, prob in dct[SRC.tok_rpr(src_word)].items():
                        y_dict[i][TRG.tok_rpr(trg_word)] += prob * alpha[i][j]
        y_dict = Variable(self._xp.array(y_dict, dtype=np.float32))
        
        # Linear Interpolation
        y = self.WD(y, y_dict)
        return y

    @staticmethod
    def _load_details(fp, args, xp, SRC, TRG):
        args.dict = defaultdict(lambda:{})
        fp.read_2leveldict(args.dict)
        args.dict = dict(args.dict)
        ChainnBasicModel._load_details(fp, args, xp, SRC, TRG)
             
    def _save_details(self, fp):
        super(DictAttentional, self)._save_details(fp)
        fp.write_2leveldict(self._dict)
    
    def report(self):
        UF.trace("W:", str(self.WD.W.data))

