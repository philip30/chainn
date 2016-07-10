import sys
from chainn.machine import ParallelTrainer
from chainn.classifier import RNNLM
from chainn.util.io import load_lm_data, batch_generator

class LMTrainer(ParallelTrainer):
    def _load_train_data(self, params):
        vocabulary, data = load_lm_data(sys.stdin, cut_threshold=params.unk_cut)
        self._src_voc   = vocabulary
        self._trg_voc   = vocabulary
        return list(batch_generator(data, (vocabulary, vocabulary), params.batch))

    def _load_dev_data(self, params):
        if params.dev:
            with open(params.dev) as dev_fp:
                # Loading Development data
                vocabulary = self._src_voc
                _, dev_data = load_lm_data(dev_fp, vocabulary)
                return list(batch_generator(dev_data, (vocabulary, vocabulary), params.batch))
    
    def _load_classifier(self, params, opt):
        return RNNLM(params, \
                X = self._src_voc, \
                Y = self._trg_voc, \
                optimizer = opt, \
                use_gpu = self.assigned_gpu, \
                collect_output = params.verbose, \
                debug_mode = params.debug)
        
    def report(self, output, src, trg, trained, epoch):
        for index in range(len(src)):
            source   = self._src_voc.str_rpr(src[index])
            ref      = self._trg_voc.str_rpr(trg[index])
            out      = self._trg_voc.str_rpr(output.y[index])
            UF.trace("Epoch (%d/%d) sample %d:\n\tINP: %s\n\tGEN: %s\n\tREF: %s" % (epoch+1, self.max_epoch, index+trained, source, out, ref))


