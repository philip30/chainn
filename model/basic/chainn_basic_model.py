import chainer.functions as F
import numpy as np

from chainer import ChainList
from chainn import Vocabulary

class ChainnBasicModel(ChainList):
    def __init__(self, src_voc, trg_voc, args, activation=F.tanh, xp=np):
        super(ChainnBasicModel, self).__init__(
            *self._construct_model(args.input, args.output, args.hidden, args.depth, args.embed)
        )
        self._input   = args.input
        self._output  = args.output
        self._hidden  = args.hidden
        self._depth   = args.depth
        self._embed   = args.embed
        self._src_voc = src_voc
        self._trg_voc = trg_voc
        self._activation = activation
        self._xp      = xp

    def save(self, fp):
        use_gpu = False
        if not self._xp == np:
            use_gpu = True
            self.to_cpu()

        fp.write(self.__class__.name)
        fp.write("Inp:\t"+str(self._input))
        fp.write("Out:\t"+str(self._output))
        fp.write("Hid:\t"+str(self._hidden))
        fp.write("Dep:\t"+str(self._depth))
        fp.write("Emb:\t"+str(self._embed))
        fp.write_activation(self._activation)
        self._src_voc.save(fp)
        self._trg_voc.save(fp)
        self._save_details(fp)
        fp.write_param_list(self)

        if use_gpu:
            self.to_gpu()
  
    @staticmethod
    def load(fp, Model, args, xp):
        args.input   = int(fp.read().split("\t")[1])
        args.output = int(fp.read().split("\t")[1])
        args.hidden = int(fp.read().split("\t")[1])
        args.depth  = int(fp.read().split("\t")[1])
        args.embed  = int(fp.read().split("\t")[1])
        act    = fp.read_activation()
        src    = Vocabulary.load(fp)
        trg    = Vocabulary.load(fp)
        Model._load_details(fp, args, xp, src, trg)
        ret    = Model(src, trg, args, act, xp)
        fp.read_param_list(ret)
        return ret

    def _construct_model(self, *args, **kwargs):
        raise NotImplementedError("Construct model is still abstract?")

    @staticmethod
    def _load_details(fp, args, xp, SRC, TRG):
        pass
    
    def _save_details(self, fp):
        pass
    
    def report(self):
        pass


