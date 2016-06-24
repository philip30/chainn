from chainn.machine import Tester
from chainn.classifier import RNN

class POSTester(Tester):
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

   
