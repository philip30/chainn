from chainer import links
from chainer.functions.evaluation import accuracy

class Classifier(links.Classifier):
    def __call__(self, x=None, t=None, update=True):
        self.y = self.predictor(x, update, t is not None)
        self.loss = self.lossfun(self.y, t)
        if self.compute_accuracy:
            self.accuracy = accuracy.accuracy(self.y, t)
        return self.loss

class NMTClassifier(links.Classifier):
    def __call__(self, x=None, t=None, update=True):
        self.y = self.predictor(x, t, update)
        self.loss = self.lossfun(self.y, t)
        if self.compute_accuracy:
            self.accuracy = accuracy.accuracy(self.y, t)
        return self.loss


