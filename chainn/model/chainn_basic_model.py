import chainer.functions as F
import numpy as np

from chainer import ChainList
from chainn import Vocabulary

class ChainnBasicModel(ChainList):
    def __init__(self, src_voc, trg_voc, args, xp=np):
        self._input   = args.input
        self._output  = args.output
        self._hidden  = args.hidden
        self._depth   = args.depth
        self._embed   = args.embed
        self._src_voc = src_voc
        self._trg_voc = trg_voc
        self._dropout = args.dropout if hasattr(args, "dropout") else 0.5
        self._xp      = xp
        super(ChainnBasicModel, self).__init__(
            *self._construct_model(args.input, args.output, args.hidden, args.depth, args.embed)
        )
            
    def _construct_model(self, *args, **kwargs):
        raise NotImplementedError("Construct model is still abstract?")

    def get_specification(self):
        ret = {\
                "input": self._input, "output": self._output, \
                "depth": self._depth, "hidden": self._hidden, \
                "embed": self._embed }
        return ret
    
    def report(self, stream, verbosity=0):
        pass

    def clean_state(self, *args, **kwargs):
        pass

    def get_state(self):
        raise NotImplementedError()
