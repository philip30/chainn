import numpy as np
import math
import ast

import chainer.functions as F
from chainer import cuda, Variable

from chainn import functions as UF
from chainn.chainer_component.functions import cross_entropy
from chainn.util.io import ModelFile

class ChainnClassifier(object):
    def __init__(self, args, X=None, Y=None, optimizer=None, use_gpu=-1, collect_output=False, activation=F.tanh):
        self._opt            = optimizer
        self._xp, use_gpu    = UF.setup_gpu(use_gpu)
        self._model          = self._load_model(args, X, Y, activation)
        self._collect_output = collect_output
        self._gpu_id         = use_gpu
        self._train_state    = self._train_state = { "loss": 150, "epoch": 0}

        if use_gpu >= 0:
            self._model = self._model.to_gpu(use_gpu)
        
        # Setup Optimizer
        if optimizer is not None:
            self._opt.setup(self._model)
    
    def train(self, x_data, y_data, learn=True, *args, **kwargs):
        accum_loss, output = self._train(x_data, y_data, not learn, *args, **kwargs)
        if learn:
            if not math.isnan(float(accum_loss.data)):
                self._model.zerograds()
                accum_loss.backward()
                self._opt.update()
            else:
                UF.trace("Warning: LOSS is nan, ignoring!")
        return accum_loss.data, output

    def classify(self, x_data, *args, **kwargs):
        return self._classify(x_data, *args, **kwargs) 

    def eval(self, x_data, y_data, *args, **kwargs):
        return self._eval(x_data, y_data, *args, **kwargs)

    def train_state(self):
        return self._train_state

    def update_state(self, epoch, loss):
        self._train_state["epoch"] = epoch
        self._train_state["loss"] = loss

    def get_vocabularies(self):
        return self._model._src_voc, self._model._trg_voc

    def _calculate_loss(self, y, ground_truth, is_train):
        return cross_entropy(y, ground_truth, cache_score=is_train)
    
    def zero(self, xp):
        return Variable(xp.zeros((), dtype=np.float32))

    def report(self):
        pass
    
    ###################
    ### Save & Load 
    ###################
    def save(self, fp):
        fp.write_optimizer_state(self._opt)
        fp.write(str(self._train_state))
        self._model.save(fp, self._gpu_id)

    def _load_model(self, args, X, Y, activation):
        assert hasattr(self, "_all_models"), "Shoudln't instantitate this class."
        
        if args.init_model:
            with ModelFile(open(args.init_model)) as model_in:
                self._opt = model_in.read_optimizer_state()
                self._train_state = ast.literal_eval(model_in.read())
                name = model_in.read()
                model = UF.select_model(name, self._all_models)
                return model.load(model_in, model, args, self._xp)
        else:
            args.input  = len(X)
            args.output = len(Y)
            return UF.select_model(args.model, self._all_models)(X, Y, args, activation=activation, xp=self._xp)
    
    ###################
    ### Abstract Method 
    ###################
    def _train(self, x_data, y_data, *args, **kwargs):
        raise NotImplementedError()

    def _decode(self, x_data, *args, **kwargs):
        raise NotImplementedError()

    def _eval(self, x_data, y_data, *args, **kwargs):
        raise NotImplementedError()

