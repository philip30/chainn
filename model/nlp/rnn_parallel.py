import numpy as np

import chainer.links as L
from chainer import Variable, cuda

from chainn import functions as UF
from chainn.model import RNN, LSTMRNN
from chainn.util import ModelFile

class RNNParallelSequence(object):
    def __init__(self, args, X=None, Y=None, optimizer=None, use_gpu=False, collect_output=False):
        self.__opt            = optimizer
        self.__xp             = cuda.cupy if use_gpu else np
        self.__model          = L.Classifier(load_model(args, X, Y))
        self.__collect_output = collect_output

        if use_gpu: self.__model = self.__model.to_gpu()
        # Setup Optimizer
        if optimizer is not None:
            self.__opt.setup(self.__model)
    
    def save(self, fp):
        self.__model.predictor.save(fp)

    def train(self, x_data, y_data, update=True):
        accum_loss, accum_acc, output = self._forward(x_data, y_data)
        if update:
            accum_loss.backward()
            accum_loss.unchain_backward()
            self.__opt.update()
        return accum_loss.data, accum_acc.data, output

    def predict(self, x_data):
        return self._forward(x_data)

    def _forward(self, x_data, y_data=None):
        xp         = self.__xp
        batch_size = len(x_data)
        src_len    = len(x_data[0])
        model      = self.__model
        is_train   = y_data is not None
    
        if is_train:
            accum_loss = 0
            accum_acc  = 0
            model.zerograds()
        
        # Forward Computation
        model.predictor.reset_state(xp, batch_size)
        output = [[] for _ in range(batch_size)]
        
        # For each word
        for j in range(src_len):
            words  = Variable(xp.array([x_data[i][j] for i in range(batch_size)], dtype=np.int32))
           
            if is_train:
                labels = Variable(xp.array([y_data[i][j] for i in range(batch_size)], dtype=np.int32))
                accum_loss += model(words, labels)
                accum_acc  += model.accuracy
            
            if not is_train or self.__collect_output:
                y = UF.argmax(model.predictor(words).data)
                for i in range(len(y)):
                    output[i].append(y[i])
        
        if is_train:
            accum_loss = accum_loss / src_len
            accum_acc  = accum_acc  / src_len
            return accum_loss, accum_acc, output
        else:
            return output

    def get_vocabularies(self):
        return self.__model.predictor._src_voc, self.__model.predictor._trg_voc

def load_model(args, X=None, Y=None):
    if args.model == "rnn":
        Model = RNN
    elif args.model == "lstm":
        Model = LSTMRNN

    if args.init_model:
        with ModelFile(open(args.init_model)) as model_in:
            return Model.load(model_in, Model)
    else:
        return Model(X, Y, embed=args.embed, \
                hidden=args.hidden, depth=args.depth, \
                input=int(1.5*len(X)), output=len(Y))

