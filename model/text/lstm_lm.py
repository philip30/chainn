
import chainer.links as L
from chainn.util import Vocabulary
from chainn.model.text import RecurrentLSTM
from chainn.link import StackLSTM

class RecurrentLSTMLM(RecurrentLSTM):
    def _save_vocabulary(self, fp):
        self._src_voc.save(fp)
   
    @staticmethod
    def _load_vocabulary(fp):
        src = Vocabulary.load(fp)
        return src, src

