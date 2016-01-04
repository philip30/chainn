
from chainer import ChainList
from chainn import Vocabulary

class ChainnBasicModel(ChainList):
    def __init__(self, *args, **kwargs):
        super(ChainnBasicModel, self).__init__(*args, **kwargs)

    def save(self, fp):
        fp.write(self.__class__.name)
        fp.write("Inp:\t"+str(self._input))
        fp.write("Out:\t"+str(self._output))
        fp.write("Hid:\t"+str(self._hidden))
        fp.write("Dep:\t"+str(self._depth))
        fp.write("Emb:\t"+str(self._embed))
        fp.write_activation(self._activation)
        self._src_voc.save(fp)
        self._trg_voc.save(fp)
        fp.write_param_list(self)
  
    @staticmethod
    def load(fp, Model, xp):
        input  = int(fp.read().split("\t")[1])
        output = int(fp.read().split("\t")[1])
        hidden = int(fp.read().split("\t")[1])
        depth  = int(fp.read().split("\t")[1])
        embed  = int(fp.read().split("\t")[1])
        act    = fp.read_activation()
        src    = Vocabulary.load(fp)
        trg    = Vocabulary.load(fp)
        ret    = Model(src, trg, input, output, hidden, depth, embed, act, xp)
        fp.read_param_list(ret)
        return ret


