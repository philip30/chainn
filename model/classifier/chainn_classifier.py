import numpy as np

import chainer.functions as F
from chainer import cuda

from chainn import functions as UF
from chainn.link import Classifier
from chainn.util import ModelFile

class ChainnClassifier(object):
    def __init__(self, args, X=None, Y=None, optimizer=None, use_gpu=False, collect_output=False, activation=F.tanh):
        self._opt            = optimizer
        self._xp             = cuda.cupy if use_gpu else np
        self._model          = self._load_classifier()(self._load_model(args, X, Y, activation))
        self._collect_output = collect_output
        self._src_voc        = X if not args.init_model else self._model.predictor._src_voc
        self._trg_voc        = Y if not args.init_model else self._model.predictor._trg_voc
        
        if use_gpu: self._model = self._model.to_gpu()
        # Setup Optimizer
        if optimizer is not None:
            self._opt.setup(self._model)
    
    def save(self, fp):
        fp.write_optimizer_state(self._opt)
        self._model.predictor.save(fp)

    def train(self, x_data, y_data, *args, update=True, **kwargs):
        accum_loss, accum_acc, output = self(x_data, y_data, *args, **kwargs)
        if kwargs.update:
            self._model.zerograds()
            accum_loss.backward()
            accum_loss.unchain_backward()
            self._opt.update()
        return accum_loss.data, accum_acc.data, output

    def get_vocabularies(self):
        return self._src_voc, self._trg_voc

    def __call__(self, x_data, y_data=None):
        raise NotImplementedError("Shouldn't instantitate this class.")

    def _load_classifier(self):
        return Classifier

    def _load_model(self, args, X, Y, activation):
        assert hasattr(self, "_all_models"), "Shoudln't instantitate this class."
        
        if args.init_model:
            with ModelFile(open(args.init_model)) as model_in:
                model_in.read_optimizer_state(self._opt)
                name = model_in.read()
                model = UF.select_model(name, self._all_models)
                return model.load(model_in, model, self._xp)
        else:
            return UF.select_model(args.model, self._all_models)(X, Y, embed=args.embed, \
                    hidden=args.hidden, depth=args.depth, \
                    input=len(X), output=len(Y), activation=activation, xp=self._xp)
 
