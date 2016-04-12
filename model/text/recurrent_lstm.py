
import chainer.links as L
from chainn.model import ChainnBasicModel
from chainn.chainer_component.links import StackLSTM

class RecurrentLSTM(ChainnBasicModel):
    name = "lstm"

    def _construct_model(self, input, output, hidden, depth, embed):
        assert(depth >= 1)
        self.embed  = L.EmbedID(input, embed)
        self.inner  = StackLSTM(embed, hidden, depth, self._dropout)
        self.output = L.Linear(hidden, output)
        return [self.embed, self.inner, self.output]

    def reset_state(self, *args, **kwargs):
        self.inner.reset_state()
    
    def __call__(self, word, ref=None, is_train=False):
        return self.output(self._activation(self.inner(self.embed(word), is_train=is_train)))
 
