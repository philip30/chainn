import numpy as np
import math
import sys
import copy
import chainer.functions as F

from collections import defaultdict
from chainer import Variable, cuda

from chainn import functions as UF
from chainn.chainer_component.functions import cross_entropy
from chainn.util.io import ModelSerializer
from chainn.model import EnsembleModel

EPSILON = 1e-6
class ChainnClassifier(object):
    def __init__(self, args, X=None, Y=None, optimizer=None, use_gpu=-1, collect_output=False, debug_mode=False):
        ## We only instantitate the descendant of this class
        assert hasattr(self, "_all_models"), "Shouldn't instantitate this class."
        ## Default configuration
        self._opt            = optimizer
        self._xp             = cuda.cupy if use_gpu >= 0 else np
        self._collect_output = collect_output
        self._gpu_id         = use_gpu
        self._train_state    = defaultdict(lambda: None, {"epoch": 0})
        self._debug_mode     = debug_mode
        self._backprop_len   = 80 if not hasattr(args, "backprop_len") else args.backprop_len
        
        ## Loading Classifier
        if args.init_model:
            if type(args.init_model) == list: # Testing time, assembly?
                self._model = EnsembleModel() # list of models for assemblying
                for i, classifier in enumerate(args.init_model):
                    serializer = ModelSerializer(classifier)
                    model = serializer.load(self, self._all_models, xp=self._xp, is_training=False)
                    UF.trace("Loaded Model %d:" % (i+1))
                    model.report(sys.stderr, verbosity=1)
                    self._model.add_model(model)
            else: # Training time, continue?
                serializer = ModelSerializer(args.init_model)
                self._model = serializer.load(self, self._all_models, xp=self._xp, is_training=True)
                self._model.report(sys.stderr, verbosity=1)
        else:
            args.input  = len(X)
            args.output = len(Y)
            self._model = UF.select_model(args.model, self._all_models)(X, Y, args, xp=self._xp)

        ## Use GPU or not?
        if use_gpu >= 0:
            self._model = self._model.to_gpu(use_gpu)

        ## Setup Optimizer
        if optimizer is not None and type(self._model) != EnsembleModel:
            self._opt.setup(self._model)
            if args.init_model:
                serializer.load_optimizer(self._opt) 
    
    #### Perfom training on parallel corpus with the specified trainer + model.
    # The method will first calculate the loss and perform backpropagation if "learn" is set to True
    def train(self, x_data, y_data, *args, learn=True, **kwargs):
        accum_loss, output = self._train(x_data, y_data, self._model, not learn, *args, **kwargs)
        if learn:
            self._back_propagate(accum_loss)
            
        return accum_loss.data, output
    
    #### Perform classification with beam search algorithm
    def classify(self, x_data, *args, gen_limit=None, beam=1, allow_empty=True, **kwargs):
        if gen_limit is None:
            gen_limit = len(x_data[0])
        # Init data
        xp         = self._xp
        TRG_len    = self._model.trg_voc_size()
        EOL        = self._model.eos_id()
        models     = self._model
        ret        = self._init_output(len(x_data), gen_limit, len(x_data[0]))
        # Init search
        queue      = [(0, self._model.get_state(), 0, None, None, None)] # (decoded_num, state, log_prob, output_word, parent, other_outputs)
        nbest_list = []
        beam_size  = min(beam, TRG_len)

        # Perform encoding + Reset state
        models.reset_state(x_data, is_train=False, *args, **kwargs)
        
        # Beam Search Decoding
        for decoded in range(gen_limit):
            # All possible states for this expansion
            expanded_states = []
            for i, state in enumerate(queue):
                _, dec_state, prob, word, _, _ = state
                model_input = self._prepare_decode_input(x_data, decoded, is_train=False)

                if beam_size <= 0 or model_input is None:
                    break
    
                if word == EOL:
                    if decoded != 1 or allow_empty:
                        nbest_list.append(state)
                        beam_size -= 1
                    continue

                if decoded != 0:
                    models.set_state(dec_state)
                    models.update(Variable(word, volatile='on'))
    
                # Conceive the next state
                # basically it will produce the prob. distribution of the target word.
                # y_o is other result
                y, y_o    = models.classify(model_input, *args, **kwargs)
                y         = y.data[0]
                dec_state = copy.deepcopy(models.get_state())
                
                # Take up several maximum words that have highest probabilities
                # Queue up the next states, with the intended update by the word for current state
                for index in UF.nargmax(y, beam_size):
                    next_word = xp.array([index], dtype=np.int32)
                    expanded_states.append(\
                            (decoded+1, dec_state, \
                            prob + math.log(y[index] + EPSILON), \
                            next_word, state, y_o))
    
            if beam_size <= 0 or len(expanded_states) == 0:
                break
            queue = sorted(expanded_states, key=lambda x: x[2], reverse=True)[:beam_size]
            
        # If the model is too bad so it doesnt produce any sentence ends with <EOL>
        if len(nbest_list) == 0:
            nbest_list = queue

        if len(nbest_list) != 0:
            sorted_nbest = sorted(nbest_list, key=lambda x:(x[2]/x[0]), reverse=True)
            state = sorted_nbest[0]
            
            # Collecting output (1-best)
            while state is not None:
                decoded, this_state, prob, word, parent, other_output = state
                if parent is not None:
                    self._collect_decoding_output(decoded-1, ret, other_output, word[0], x_data[0])
                    
                # prev_state state
                state = parent
        return ret
    
    #### Report the internal model current state
    def report(self):
        if type(self._model) == list:
            for model in self._model:
                model.predictor.report()
        else:
            self._model.predictor.report()

    def _back_propagate(self, accum_loss):
        if not self._debug_mode or not math.isnan(float(accum_loss.data)):
            self._model.zerograds()
            accum_loss.backward()
            accum_loss.unchain_backward()
            self._opt.update()
        else:
            UF.trace("Warning: LOSS is nan, ignoring!")

    #######################
    ### Setter + Getter ###
    #######################
    def get_train_state(self):
        return self._train_state
    
    def get_vocabularies(self):
        if type(self._model) == EnsembleModel:
            return self._model[0]._src_voc, self._model[0]._trg_voc
        else:
            return self._model._src_voc, self._model._trg_voc
    
    def get_specification(self):
        return self._train_state
    
    def update_state(self, loss, epoch, dev_loss=None):
        self._train_state["loss"] = loss
        self._epoch["epoch"] = epoch

        if dev_loss is not None:
            self._dev_loss["dev_loss"] = dev_loss
        

    def set_specification(self, spec):
        self._train_state = spec

    ###################
    ### Protected method
    ###################
    def _train(self, x_data, y_data, model, is_dev, *args, **kwargs):
        xp         = self._xp
        src_len    = len(x_data[0])
        batch_size = len(x_data)
        gen_limit  = len(y_data[0])
        is_train   = not is_dev
        volatile   = "off" if is_train else "on"
        
        if self._collect_output:
            output = self._init_output(batch_size, gen_limit, src_len)
        else:
            output = None

        # Perform encode target side
        model.reset_state(x_data, *args, is_train=is_train, **kwargs)

        # Perform decode 
        accum_loss = 0
        for j in range(gen_limit):
            # Input word
            model_input = self._prepare_decode_input(x_data, j, is_train=is_train)
            # Target Word
            s_t         = Variable(xp.array([y_data[i][j] for i in range(len(y_data))], dtype=np.int32), volatile=volatile)
            # Execute model to get target's probability distribution
            doutput     = model(model_input, *args, is_train=is_train, **kwargs)
            # Accumulate the loss
            accum_loss += self._calculate_loss(doutput.y, s_t, is_dev)
            # Update the model state
            model.update(s_t, is_train=is_train)

            # Check for bptt len
            if (j+1) % self._backprop_len == 0:
                self._back_propagate(accum_loss)

            # Collecting output
            if self._collect_output:
                self._collect_decoding_output(j, output, doutput)
        # Clean all the states 
        model.clean_state()
        
        return (accum_loss / gen_limit), output
    
    def _calculate_loss(self, y, ground_truth, is_train):
        return cross_entropy(y, ground_truth, cache_score=is_train)

    ##############################
    ### Abstract Method
    ##############################
    def _collect_decoding_output(self, time_step, holder, out, word=None, src=None):
        raise NotImplementedError()

    def _prepare_decode_input(self, input_batch, time_step, is_train=True):
        raise NotImplementedError()

