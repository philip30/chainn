import chainer.links as L
import chainer.functions as F
from chainer import ChainList

class StackLSTM(ChainList):
    def __init__(self, I, O, depth, drop_ratio=0.0):
        chain_list = []
        for i in range(depth):
            start = I if i == 0 else O
            chain_list.append(L.LSTM(start, O))
        self._drop_ratio = drop_ratio
        super(StackLSTM, self).__init__(*chain_list)
    
    def reset_state(self):
        for lstm in self:
            lstm.reset_state()
    
    def __call__(self, inp, is_train=False):
        ret = None
        for i, hidden in enumerate(self):
            h = inp if i == 0 else ret
            ret = hidden(h)
        return F.dropout(ret, train=is_train, ratio=self._drop_ratio)
    
    def get_state(self):
        ret = []
        for lstm in self:
            ret.append((lstm.c, lstm.h))
        return ret

    def set_state(self, state):
        for lstm_self, lstm_in in zip(self, state):
            lstm_self.c, lstm_self.h = lstm_in

