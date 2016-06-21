
import random
import numpy as np

from chainer import cuda

class ParallelTrainer(object):
    def __init__(self, seed=0, use_gpu=-1):
        init_gpu(use_gpu)
        init_seed(seed, use_gpu)

    # training from parallel corpus
    def train(self, train_data, model, max_epoch, \
            onEpochStart, onBatchUpdate, onEpochUpdate, onTrainingFinish, one_epoch=False):
        train_state   = model.get_train_state()
       
        # If we are starting from middle, make sure that we have the same randomness
        # use the same seed by --seed
        for _ in range(train_state["epoch"]):
            random.shuffle(train_data)
        
        # Iteration starts here
        for epoch in range(train_state["epoch"], max_epoch):
            trained        = 0
            epoch_loss     = 0
            
            # Shuffling batch
            random.shuffle(train_data)

            # Training from the corpus
            ## TODO count word per second
            onEpochStart(epoch)
            for src, trg in train_data:
                accum_loss, output = model.train(src, trg)
                epoch_loss        += accum_loss
                
                trained += len(src)
                onBatchUpdate(output, src, trg, trained, epoch, accum_loss)
                            
            # Normalize the loss divided by len of the batches
            epoch_loss /= len(train_data)

            # Print the epoch report
            onEpochUpdate(epoch_loss, train_state["loss"], epoch)
            
            # Keep track of #iteration and #loss of the model
            model.update_state(loss=float(epoch_loss), epoch=epoch+1)
            
            # This is for one epoch training
            # The system will terminate the training after one epoch is finished.
            # This is useful for epoch-to-epoch evaluation.
            # specify the maximum number of iteration by --epoch and use --one_epoch --model_out model_n 
            # to stop the training. 
            # Afte that initialize the model by --init_model model_n
            if one_epoch:
                break

        onTrainingFinish(train_state["epoch"])

    # Evaluation on development set
    def eval(self, dev_data, classifier):
        epoch_loss = 0
        for src, trg in dev_data:
            loss, _ = classifier.train(src, trg, learn=False)
            epoch_loss += float(loss)
        return epoch_loss / len(dev_data)

### Helper functions
# Initialize seed for both CPU and GPU
def init_seed(seed, use_gpu):
    if seed != 0:
        np.random.seed(seed)
        if use_gpu >= 0 and hasattr(cuda, "cupy"):
            cuda.cupy.random.seed(seed)
        random.seed(seed)

# Select the GPU to be used
def init_gpu(use_gpu):
    if hasattr(cuda, "cupy"):
        if use_gpu >= 0:
            cuda.get_device(use_gpu).use()


