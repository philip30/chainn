import sys
from chainn.machine import ParallelTrainer
from chainn.classifier import RNN
from chainn.util.io import load_pos_train_data, batch_generator

class POSTrainer(ParallelTrainer):
    def _load_train_data(self, params):
        src_voc, trg_voc, data = load_pos_train_data(sys.stdin, cut_threshold=params.unk_cut)
        self._src_voc = src_voc
        self._trg_voc = trg_voc
        return list(batch_generator(data, (src_voc, trg_voc), params.batch))
    
    def _load_dev_data(self, params):
        if params.dev:
            with open(params.dev) as dev_fp:
                # Loading Development data
                src_voc = self._src_voc
                trg_voc = self._trg_voc
                _, _, dev_data = load_pos_train_data(dev_fp, src_voc, trg_voc)
                return list(batch_generator(dev_data, (src_voc, trg_voc), params.batch))
    
    def _load_classifier(self, params, opt):
        return RNN(params, \
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
            UF.trace("Epoch (%d/%d) sample %d:\n\tINP: %s\n\tTAG: %s\n\tREF: %s" % (epoch+1, self.max_epoch, index+trained, source, out, ref))


