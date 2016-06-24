import sys
from chainn.util import AlignmentVisualizer, functions as UF
from chainn.machine import Tester
from chainn.classifier import EncDecNMT

class NMTTester(Tester):
    def __init__(self, params, loader):
        super(NMTTester, self).__init__(params, loader)
        self.align_stream = UF.load_stream(params.align_out)

    def load_classifier(self, params):
        model = EncDecNMT(params, use_gpu=self.assigned_gpu, collect_output=True)
        src_voc, trg_voc = model.get_vocabularies()
        self._src_voc = src_voc
        self._trg_voc = trg_voc
        return model
    
    def onSingleUpdate(self, ctr, src, trg):
        align_fp = self.align_stream if self.align_stream is not None else sys.stderr
        print(self._trg_voc.str_rpr(trg.y[0]))
        
        if trg.a is not None:
            AlignmentVisualizer.print(trg.a, ctr, src, trg.y, \
                    self._src_voc, self._trg_voc, fp=align_fp)

    def onDecodingFinish(self):
        if self.align_stream is not None:
            self.align_stream.close()
        UF.trace("Finished decoding")

    def load_decoding_options(self, params):
        return {"gen_limit": params.gen_limit, "eos_disc": params.eos_disc, "beam": params.beam}

   
