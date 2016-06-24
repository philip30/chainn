import chainer.functions as F

class EnsembleModel(object):
    def __init__(self, ensemble_method="linear"):
        self._models = []
        self._method = ensemble_method

    def add_model(self, model):
        self._models.append(model)

    def get_state(self):
        return [model.get_state() for model in self._models]

    def to_gpu(self, use_gpu):
        for model in self._models:
            model.to_gpu(use_gpu)
        return self
    
    def eos_id(self):
        return self._models[0]._trg_voc.eos_id()
    
    def trg_voc_size(self):
        return len(self._models[0]._trg_voc)

    def reset_state(self, x_data, *args, is_train=False, **kwargs):
        for model in self._models:
            model.reset_state(x_data, *args, is_train=is_train, **kwargs)
    
    def set_state(self, states):
        for model, state in zip(self._models, states):
            model.set_state(state)
    
    def update(self, word):
        for model in self._models:
            model.update(word, is_train=False)
    
    def classify(self, src, *args, **kwargs):
        if self._method == "linear":
            y = 0
            normalizer = len(self._models)
            for i, model in enumerate(self._models):
                doutput = model(src, *args, is_train=False, **kwargs)
                y       += doutput.y

                # Here we only store the first full result of all the models
                if i == 0:
                    other_results = doutput
            y /= normalizer
        else:
            raise NotImplementedError()
        return y, other_results
    
    def __getitem__(self, index):
        return self._models[index]
