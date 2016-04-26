
import gc
import random
import numpy as np

from chainer import cuda

def init_seed(seed):
    if seed != 0:
        np.random.seed(seed)
        if hasattr(cuda, "cupy"):
            cuda.cupy.random.seed(seed)
        random.seed(seed)

def init_gpu(use_gpu):
    if hasattr(cuda, "cupy"):
        if use_gpu >= 0:
            cuda.get_device(use_gpu).use()

class ParallelTrainer:
    def __init__(self, seed=0, use_gpu=-1):
        init_gpu(use_gpu)
        init_seed(seed)

    def train(self, train_data, model, max_epoch, \
            onEpochStart, onBatchUpdate, onEpochUpdate, onTrainingFinish, one_epoch=False):
        train_state = model.train_state()
        prev_loss   = train_state["loss"]
        trained_epoch = -1
        for epoch in range(train_state["epoch"], max_epoch):
            trained        = 0
            epoch_loss     = 0
            
            # Shuffling batch
            random.shuffle(train_data)

            # Training from the corpus
            onEpochStart(epoch)
            for src, trg in train_data:
                accum_loss, output = model.train(src, trg)
                epoch_loss     += accum_loss
                
                trained += len(src)
                onBatchUpdate(output, src, trg, trained, epoch, accum_loss)
                            
            # Cleaning up fro the next epoch
            epoch_loss /= len(train_data)
            model.update_state(loss=float(epoch_loss), epoch=epoch+1)
            onEpochUpdate(epoch_loss, prev_loss, epoch)
            prev_loss = epoch_loss
            gc.collect()
            trained_epoch = epoch
            if one_epoch:
                break

        onTrainingFinish(trained_epoch)

    def eval(self, dev_data, model):
        accum_loss = 0
        for src, trg in dev_data:
            loss, _ = model.train(src, trg, learn=False)
            accum_loss += float(loss)
            gc.collect()
        return accum_loss / len(dev_data)

