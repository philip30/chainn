import numpy as np
import util.functions as UF

from chainer import optimizers, cuda, Variable
from collections import defaultdict

class NMT:
    """ 
    Constructor 
    """
    def __init__(self, src_voc=None, trg_voc=None,\
            optimizer=optimizers.SGD(), gc=10, hidden=5, \
            embed=5, input=5, output=5, compile=True, use_gpu=False,
            gen_limit=50, dictionary=None):
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
        self._dict = dictionary
        if compile:
            self._model = self._init_model()
            
            if dictionary is not None:
                self._dict = self._load_dictionary(dictionary)

    """ 
    Publics 
    """ 
    def init_params(self):
        if self._use_gpu: self._model.to_cpu()
        UF.init_model_parameters(self._model, -0.08, 0.08, seed=1) # special seed number
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

    def _dictionary_consideration(self, src, y, alpha):
        SRC = self._src_voc
        TRG = self._trg_voc
        dct = self._dict
        xp  = self._xp
#        alpha_max = np.argmax(np.array(alpha), axis=1)
        
        # No dictionary is specified
        if self._dict is None:
            return y
        score = xp.zeros((len(src), self._output), dtype=np.float32)
        for src_i, sent in enumerate(src):
            for word_i, src_word in enumerate(sent):
                for trg_word, dct_prob in dct[src_word].items():
                    score[src_i][trg_word] += alpha[src_i][word_i] * dct_prob
#            a_max = alpha_max[src_i]
#            print(a_max)
#            for trg_word, dct_prob in dct[a_max].items():
#                score[a_max][trg_word] += alpha[src_i][a_max] * dct_prob

#        print(SRC)
#        print(TRG)
#        print("+++SCORE+++")
#        print(score)
#        print("+++Y+++")
#        print(y.data)
        score = Variable(score)
        ret = y + score
#        print("+++RET+++")
#        print(ret.data)
        return ret

    def _load_dictionary(self, dict_dir):
        dct = defaultdict(lambda:defaultdict(lambda: 0))
        SRC = self._src_voc
        TRG = self._trg_voc
        with open(dict_dir) as fp:
            for line in fp:
                line = line.strip().split()
                dct[SRC[line[0]]][TRG[line[1]]] = float(line[2])
        return dct

