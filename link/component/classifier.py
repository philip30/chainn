from chainer import links
from chainer.functions.evaluation import accuracy

class Classifier(links.Classifier):
    def __call__(self, x=None, t=None, update=True):
        self.y = self.predictor(x, update, t is not None)
        self.loss = self.lossfun(self.y, t)
        return self.loss

class NMTClassifier(links.Classifier):
    def __call__(self, x=None, t=None, *args, **kwargs):
        self.output = self.predictor(x, t, *args, **kwargs)
        self.y = self.output.y
        if t is not None:
            self.loss = self.lossfun(self.y, t)
            return self.loss
        else:
            return self.output


