import chainer.links as L
import chainer.functions as F
import copy
from chainer import Variable, ChainList
from chainn.link import LSTM

class StackLSTM(ChainList):
    
    def __init__(self, I, O, depth):
        chain_list = []
        for i in range(depth):
            start = I if i == 0 else O
            chain_list.append(LSTM(start, O))
        super(StackLSTM, self).__init__(*chain_list)

    def __call__(self, inp, is_train=False):
        ret = None
        for i, lstm in enumerate(self):
            h = inp if i == 0 else ret
            ret = lstm(h)
            ret = F.dropout(ret, train=is_train, ratio=0.2)
        return ret
    
    def reset_state(self):
        for lstm in self:
            lstm.reset_state()

    def copy_state(self, other):
        for lstm_1, lstm_2 in zip(self, other):
            lstm_1.c = copy.copy(lstm_2.c)

