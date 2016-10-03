from chainn.classifier import RNN
from chainn.util.io import load_pos_test_data
from chainn.machine import Tester

class POSTester(Tester):
    def __init__(self, param):
        super(POSTester, self).__init__(param, load_pos_test_data)

    def load_classifier(self, params):
        model = RNN(params, use_gpu=self.assigned_gpu, collect_output=True)
        src_voc, trg_voc = model.get_vocabularies()
        self._src_voc = src_voc
        self._trg_voc = trg_voc
        return model
    
    def onSingleUpdate(self, ctr, src, trg):
        print(self._trg_voc.str_rpr(trg.y[0]))

    def load_decoding_options(self, params):
        return { "beam": params.beam }

