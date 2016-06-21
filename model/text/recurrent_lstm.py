
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import Variable
from chainn.model import ChainnBasicModel
from chainn.util import DecodingOutput
from chainn.chainer_component.links import StackLSTM

class RecurrentLSTM(ChainnBasicModel):
    name = "lstm"

    def _construct_model(self, input, output, hidden, depth, embed):
        assert(depth >= 1)
        self.embed  = L.EmbedID(input, embed)
        self.inner  = StackLSTM(embed, hidden, depth, self._dropout)
        self.output = L.Linear(hidden, output)
        return [self.embed, self.inner, self.output]

    def reset_state(self, x_data, is_train=False, *args, **kwargs):
        self.inner.reset_state()
        volatile = "off" if is_train else "on"
        self.h = Variable(self._xp.zeros((len(x_data), self._hidden), dtype=np.float32), volatile=volatile)
    
    def __call__(self, word, ref=None, is_train=False, *args, **kwargs):
        return DecodingOutput({"y": F.softmax(self.output(self.h))})
    
    def update(self, word, is_train=False):
        self.h = F.tanh(self.inner(self.embed(word), is_train=is_train))
    
    def get_state(self):
        return self.inner.get_state()

    def set_state(self, state):
        self.inner.set_state(state)

