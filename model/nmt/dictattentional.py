import numpy as np
import sys
from functools import reduce
import chainer.functions as F
import chainer.links as L

# Chainer
from chainer import Variable

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
        SRC = self._src_voc
        TRG = self._trg_voc
        dct = self._xp.zeros((len(SRC), len(TRG)), dtype=np.float32)
        with open(dict_dir) as fp:
            for line in fp:
                line = line.strip().split()
                if line[0] in SRC and line[1] in TRG:
                    dct[SRC[line[0]]][TRG[line[1]]] = float(line[2])
        return dct

    def _additional_score(self, y, a, src):
        SRC = self._src_voc
        TRG = self._trg_voc
        dct = self._dict
        xp  = self._xp
        dct = Variable(self._dict)
        batch_size = len(y)
        if len(a) > 0:
            alpha = a[0]
            for i in range(1, len(a)):
                alpha = F.concat((alpha, a[i]), axis=1)
            
            y_dict = None
            for batch_id, batch in enumerate(src):
                alpha_v = xp.zeros((len(SRC),1), dtype=np.float32)
                for src_i, src_word in enumerate(batch):
                    alpha_v[src_word][0] = alpha.data[batch_id][src_i]
                alpha_v = Variable(alpha_v)
                y_dict_n = F.matmul(alpha_v, dct, transa=True)
                if y_dict is None:
                    y_dict = y_dict_n
                else:
                    y_dict = F.concat((y_dict, y_dict_n), axis=0)
            y = self.WD(y_dict, y)

        return y

    @staticmethod
    def _load_details(fp, args, xp, SRC, TRG):
        args.dict = xp.zeros((len(SRC), len(TRG)), dtype=np.float32)
        fp.read_matrix(args.dict, float)
        ChainnBasicModel._load_details(fp, args, xp, SRC, TRG)
             
    def _save_details(self, fp):
        super(DictAttentional, self)._save_details(fp)
        fp.write_matrix(self._dict)

