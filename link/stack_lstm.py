import chainer.links as L
from chainer import Variable, ChainList
from chainn.link import LSTM

class StackLSTM(ChainList):
    
    def __init__(self, I, O, depth):
        chain_list = []
        for i in range(depth):
            start = I if i == 0 else O
            chain_list.append(LSTM(start, O))
        super(StackLSTM, self).__init__(*chain_list)

    def __call__(self, inp):
        ret = None
        for i, lstm in enumerate(self):
            h = inp if i == 0 else ret
            ret = lstm(h)
        return ret
    
    def reset_state(self):
        for lstm in self:
            lstm.reset_state()