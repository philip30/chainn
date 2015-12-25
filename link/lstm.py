import chainer.links as L
from chainer import variable
from chainer.functions.activation import lstm

class LSTM(L.LSTM):
    def __call__(self, x, update=True):
        lstm_in = self.upward(x)
        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        c, h = lstm.lstm(self.c, lstm_in)
        if update:
            self.c, self.h = c, h
        return h

