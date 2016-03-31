import numpy as np
import math
from collections import defaultdict

# Chainer
from chainer import Variable, cuda
import chainer.functions as F

# Chainn
import chainn.util.functions as UF
from chainn.classifier import ChainnClassifier
from chainn.model.nmt import EncoderDecoder, Attentional, DictAttentional
from chainn.util import DecodingOutput
from chainn.util.io import ModelFile

class EncDecNMT(ChainnClassifier):
    def __init__(self, *args, **kwargs):
        self._all_models = [EncoderDecoder, Attentional, DictAttentional]
        super(EncDecNMT, self).__init__(*args, **kwargs)

    def _train(self, x_data, y_data, is_dev, *args, **kwargs):
        xp         = self._xp
        src_len    = len(x_data[0])
        batch_size = len(x_data)
        gen_limit  = len(y_data[0])
        is_train   = not is_dev
        
        # Perform encoding + Reset state
        self._model.reset_state(x_data, is_train=is_train, *args, **kwargs)

        if self._collect_output:
            output    = np.zeros((batch_size, gen_limit), dtype=np.float32)
            alignment = np.zeros((batch_size, gen_limit, src_len), dtype=np.float32)
        
        accum_loss = 0
        # Decoding
        for j in range(gen_limit):
            s_t         = Variable(xp.array([y_data[i][j] for i in range(len(y_data))], dtype=np.int32))
            doutput     = self._model(x_data, is_train=is_train, *args, **kwargs) # Decode one step
            accum_loss += self._calculate_loss(doutput.y, s_t)
            
            self._model.update(self._select_update(doutput.y, s_t, is_train=is_train), is_train=is_train)

            # Collecting output
            if self._collect_output:
                collect_output(j, output, alignment, doutput)
            if doutput.a is None:
                alignment = None

        self._model.clean_state()
        
        output = DecodingOutput(output, alignment) if self._collect_output else None
        return (accum_loss / gen_limit), output

    def _classify(self, x_data, gen_limit=50, beam=1, beam_pick=10, *args, **kwargs):
        # Unpacking
        xp         = self._xp
        batch_size = len(x_data)
        src_len    = len(x_data[0])
        EOL        = self._model._trg_voc.eos_id()
        
        # Perform encoding + Reset state
        self._model.reset_state(x_data, is_train=False, *args, **kwargs)

        output    = np.zeros((batch_size, gen_limit), dtype=np.float32)
        alignment = np.zeros((batch_size, gen_limit, src_len), dtype=np.float32)
        if beam == 1:
            # Normal Decoding
            for j in range(gen_limit):
                doutput = self._model(x_data, is_train=False, *args, **kwargs)
                
                self._model.update(self._select_update(doutput.y, None, is_train=False), is_train=False)
    
                if doutput.a is None:
                    alignment = None
    
                # Collecting output
                collect_output(j, output, alignment, doutput)
                if all(output[i][j] == EOL for i in range(len(output))):
                    break
        else:
            model_update = lambda x: self._model.update(Variable(self._xp.array([[x]], dtype=np.int32)), is_train=False)
            # Beam Decoding
            beam_search(beam, beam_pick, gen_limit, self._model, x_data, output, alignment, model_update)

        self._model.clean_state()
        
        return DecodingOutput(output, alignment)

    def _calculate_loss(self, y, ground_truth):
        return F.softmax_cross_entropy(y, ground_truth)

    # Update the RNN state 
    def _select_update(self, y, train_ref, is_train):
        if train_ref is not None and is_train:
            # Training
            wt = train_ref
        else:
            # Testing
            wt = Variable(self._xp.array(UF.argmax(y.data), dtype=np.int32))
        return wt

    def report(self):
        self._model.predictor.report()

eps = 1e-6
def beam_search(beam_size, beam_pick, gen_limit, model, src, output, alignment, model_update, *args, **kwargs):
    queue     = [(0, model.get_state(), 0, DecodingOutput(), None)] # (num_decoded, state, log prob, output, parent)
    take_size = beam_pick                              # Only take top 10 words
    loop      = 0
    EOL       = model._trg_voc.eos_id()
    queue_state = None
    # Beam search algorithm
    while type(loop) == int:
        queue_state = queue.pop(0)
        decoded, state, prob, out, parent = queue_state
        if out.y == EOL or decoded == gen_limit:
            break
        
        # Setting up for decoding
        if loop != 0:
            model.set_state(state)
            model_update(out.y)

        # Conceive the next state
        doutput = model(src, is_train=False, *args, **kwargs)
        # Take the best 1 (beam search only for single decoding)
        y     = F.softmax(doutput.y).data[0]
        out_a = doutput.a.data[0]
        # Take up several words that makes maximum
        agmx = UF.argmax_index(y, take_size)
        # Update model state based on those words
        for index in agmx:
            queue.append((decoded+1, state, prob + math.log(y[index] + eps), DecodingOutput(index, out_a), queue_state))

        queue = sorted(queue, key=lambda x: x[2], reverse=True)
        
        queue = queue[:min(len(queue), beam_size)+1]
        loop = 1

    # Collecting output (1-best)
    while queue_state is not None:
        decoded, state, prob, out, parent = queue_state
        decoded-=1
        if parent is not None:
            output[0][decoded] = out.y
            if out.a is not None:
                for i, x_a in enumerate(out.a):
                    alignment[0][decoded][i] = float(x_a)
            
        # Next state
        queue_state = parent
    return output, alignment

def collect_output(src_col, output, alignment, out):
    y = UF.argmax(out.y.data)
    for i in range(len(y)):
        output[i][src_col] = y[i]
    
    a = out.a
    if a is not None:
        for i, x in enumerate(a.data):
            for j, x_a in enumerate(x):
                alignment[i][src_col][j] = float(x_a)

