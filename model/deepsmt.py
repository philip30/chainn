import numpy as np
import util.functions as UF

from chainer import optimizers, cuda

class DeepSMT:
    """ 
    Constructor 
    """
    def __init__(self, src_voc=None, trg_voc=None,\
            optimizer=optimizers.SGD(), gc=10, hidden=5, \
            embed=5, input=5, output=5, compile=True, use_gpu=False,
            gen_limit=50):
        self._optimizer = optimizer
        self._gc = gc
        self._hidden = hidden
        self._embed = embed
        self._input = input
        self._output = output
        self._src_voc = src_voc
        self._trg_voc = trg_voc
        self._use_gpu = use_gpu
        self._gen_lim = gen_limit
        self._xp = cuda.cupy if use_gpu else np
        if compile:
            self._model = self._init_model()
        
    """ 
    Publics 
    """ 
    def init_params(self):
        if self._use_gpu: self._model.to_cpu()
        UF.init_model_parameters(self._model, -0.08, 0.08)
        if self._use_gpu: self._model.to_gpu()
  
    def setup_optimizer(self):
        self._optimizer.setup(self._model)

    def train(self, src_batch, trg_batch):
        return self._forward_training(src_batch, trg_batch)
    
    def decay_lr(self, decay_factor):
        self._optimizer.lr /= decay_factor
    
    def update(self, loss):
        self._optimizer.zero_grads()
        loss.backward()
        loss.unchain_backward()
        self._optimizer.clip_grads(self._gc)
        self._optimizer.update()
    
    def decode(self, src_batch):
        return self._forward_testing(src_batch)

    def save(self, fp):
        raise NotImplementedError()

    def load(self, fp):
        raise NotImplementedError()

    def get_vocabularies(self):
        return self._src_voc, self._trg_voc

    """
    Privates
    """
    def _init_model(self):
        model = self._construct_model()
        return model.to_gpu() if self._use_gpu else model

    def _construct_model(self):
        raise NotImplementedError()

    def _forward_training(self, src_batch, trg_batch):
        raise NotImplementedError()

    def _forward_testing(self, src_batch):
        raise NotImplementedError()

