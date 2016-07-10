import sys, random, math
import numpy as np

from chainer import cuda, optimizers
from chainn.util import functions as UF
from chainn.util.io import ModelSerializer

class ParallelTrainer(object):
    def __init__(self, params):
        self.assigned_gpu = UF.init_global_environment(params.seed, params.gpu, params.use_cpu)
        self.save_models  = params.save_models
        self.verbose      = params.verbose
        self.model_out    = params.model_out
        self.max_epoch    = params.epoch
        self.one_epoch    = True if params.one_epoch else False
        self.train_data   = self._load_train_data(params)
        self.dev_data     = self._load_dev_data(params)
        self.classifier   = self._load_classifier(params, select_optimizer(params.optimizer))
    
    # training from parallel corpus
    def train(self):
        model         = self.classifier
        max_epoch     = self.max_epoch
        train_data    = self.train_data
        dev_data      = self.dev_data
        continue_next = True
        train_state   = model.get_train_state()
       
        # If we are starting from middle, make sure that we have the same randomness
        # use the same seed by --seed
        for _ in range(train_state["epoch"]):
            random.shuffle(train_data)
        
        # Iteration starts here
        self.onTrainingStart(train_state)
        for epoch in range(train_state["epoch"], max_epoch):
            trained        = 0
            epoch_loss     = 0
            
            # Shuffling batch
            random.shuffle(train_data)

            # Training from the corpus
            ## TODO count word per second
            self.onEpochStart(epoch)
            for src, trg in train_data:
                accum_loss, output = model.train(src, trg)
                epoch_loss        += accum_loss
                                
                trained += len(src)
                
                self.onBatchUpdate(output, src, trg, trained, epoch, accum_loss)
                            
            # Normalize the loss divided by len of the batches
            epoch_loss /= len(train_data)
            # Print the epoch report
            continue_next = self.onEpochUpdate(epoch_loss, train_state, epoch)

            if not continue_next:
                break

            # This is for one epoch training
            # The system will terminate the training after one epoch is finished.
            # This is useful for epoch-to-epoch evaluation.
            # specify the maximum number of iteration by --epoch and use --one_epoch --model_out model_n 
            # to stop the training. 
            # After that initialize the model by --init_model model_n
            if self.one_epoch:
                break
       
        training_finish = train_state["epoch"] == max_epoch or not continue_next
        self.onTrainingFinish(train_state["epoch"], training_finish)

    # Evaluation on development set
    def eval(self):
        epoch_loss = 0
        num_batch  = len(self.dev_data)
        for src, trg in self.dev_data:
            loss, _ = self.classifier.train(src, trg, learn=False)
            epoch_loss += float(loss)
        return epoch_loss / num_batch
       
    # Callback
    def onTrainingStart(self, train_state):
        pass

    def onEpochStart(self, epoch):
        UF.trace("Starting Epoch", epoch+1)
    
    def onEpochUpdate(self, epoch_loss, prev_state, epoch):
        # Reporting Perplexity of training data
        ppl       = math.exp(epoch_loss)
        if prev_state["loss"] is not None:
            prev_ppl  = math.exp(prev_state["loss"])
            UF.trace("Train PPL:", prev_ppl, "->", ppl)
        else:
            UF.trace("Train PPL:", ppl)
        prev_state["loss"] = float(epoch_loss)
    
        # Reporting Perplexity of testing data
        # If perplexity increased, stop the iteration earlier
        continue_next_iter    = True
        if self.dev_data is not None:
            prev_dev_ppl = None
            dev_loss = self.eval()
            dev_ppl  = math.exp(dev_loss)
            if "dev_loss" in prev_state:
                prev_dev_ppl = math.exp(prev_state["dev_loss"])
                
                if dev_ppl >= prev_dev_ppl:
                    continue_next_iter = False

                UF.trace("Dev PPL:", prev_dev_ppl, "->", dev_ppl)
            else:
                UF.trace("Dev PPL:", dev_ppl)
            prev_state["dev_loss"] = float(dev_loss)
        
        # Saving model if only perplexity is not increased
        if continue_next_iter:
            self.save_classifier(epoch)
        else:
            UF.trace("Development perplexity increased, finishing iteration now.")
        
        # Updating epoch information
        prev_state["epoch"] += 1
        return continue_next_iter
    
    def onBatchUpdate(self, output, src, trg, trained, epoch, accum_loss):
        if self.verbose:
            self.report(output, src, trg, trained, epoch)
        UF.trace("Trained %d, PPL=%f, col_size=%d" % (trained, math.exp(accum_loss), len(trg[0])-1)) # minus the last </s>

    def onTrainingFinish(self, epoch, training_finished):
        if not self.save_models and not training_finished:
            self.save_classifier(epoch)
        if training_finished:
            UF.trace("Training complete!")

    def report(self, output, src, trg, trained, epoch):
        pass

    def print_details(self):
        pass

    def save_classifier(self, epoch):
        out_file = self.model_out
        if self.save_models:
            out_file += "-" + str(epoch)
        UF.trace("saving model to " + out_file + "...")
        serializer = ModelSerializer(out_file)
        serializer.save(self.classifier)

    #######################
    # Abstract Method
    #######################
    def _load_train_data(self, params):
        raise NotImplementedError()
    
    def _load_dev_data(self, params):
        raise NotImplementedError()

    def _load_classifier(self, params):
        raise NotImplementedError()

# TODO: More Optimizer
def select_optimizer(opt_string):
    return optimizers.Adam()

