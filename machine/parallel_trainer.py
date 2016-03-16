
import gc
import random
import numpy as np

from chainer import cuda
from chainn.util.io import batch_generator

def init_seed(seed):
    if seed != 0:
        np.random.seed(seed)
        if hasattr(cuda, "cupy"):
            cuda.cupy.random.seed(seed)
        random.seed(seed)

class ParallelTrainer:
    def __init__(self, seed=0):
        init_seed(seed)

    def load_data(self, src, trg, loader, batch, cut):
        with open(src) as src_fp:
            with open(trg) as trg_fp:
                SRC, TRG, data = loader(src_fp, trg_fp, cut_threshold=cut)
        return SRC, TRG, list(batch_generator(data, (SRC, TRG), batch_size=batch))

    def train(self, train_data, model, max_epoch, \
            onEpochStart, onBatchUpdate, onEpochUpdate, onTrainingFinish):
        prev_loss  = 150
        for epoch in range(max_epoch):
            trained        = 0
            epoch_loss     = 0
            epoch_accuracy = 0
            
            # Shuffling batch
            random.shuffle(train_data)

            # Training from the corpus
            onEpochStart(epoch)
            for src, trg in train_data:
                accum_loss, accum_acc, output = model.train(src, trg)
                epoch_loss     += accum_loss
                epoch_accuracy += accum_acc

                onBatchUpdate(output, src, trg, trained, epoch, accum_loss)
                trained += len(src)
            epoch_loss     /= len(train_data)
            epoch_accuracy /= len(train_data)
            
            # Cleaning up fro the next epoch
            onEpochUpdate(epoch_loss, epoch_accuracy, prev_loss, epoch)
            prev_loss = epoch_loss
            gc.collect()

        onTrainingFinish(epoch)
    
