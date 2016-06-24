from chainn.machine import ParallelTrainer
from chainn.classifier import EncDecNMT
from chainn.util.io import load_nmt_train_data, batch_generator

class NMTTrainer(ParallelTrainer):
    def _load_train_data(self, params):
        with open(params.src) as src_fp:
            with open(params.trg) as trg_fp:
                src_voc, trg_voc, train_data = load_nmt_train_data(src_fp, trg_fp, cut_threshold=params.unk_cut)
                self._src_voc = src_voc
                self._trg_voc = trg_voc
                return list(batch_generator(train_data, (src_voc, trg_voc), params.batch))
    
    def _load_dev_data(self, params):
        if params.src_dev and params.trg_dev:
            with open(params.src_dev) as src_fp:
                with open(params.trg_dev) as trg_fp:
                    # Loading Development data
                    src_voc = self._src_voc
                    trg_voc = self._trg_voc
                    _, _, dev_data = load_nmt_train_data(src_fp, \
                            trg_fp, src_voc, trg_voc)
                    return list(batch_generator(dev_data, (src_voc, trg_voc), params.batch))
    
    def _load_classifier(self, params, opt):
        return EncDecNMT(params, \
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
            UF.trace("Epoch (%d/%d) sample %d:\n\tSRC: %s\n\tOUT: %s\n\tREF: %s" % (epoch+1, self.max_epoch, index+trained, source, out, ref))


