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
        
        accum_loss = self.zero(xp)
        # Decoding
        for j in range(gen_limit):
            s_t         = Variable(xp.array([y_data[i][j] for i in range(len(y_data))], dtype=np.int32))
            doutput     = self._model(x_data, is_train=is_train, *args, **kwargs) # Decode one step
            accum_loss += self._calculate_loss(doutput.y, s_t, is_dev)
            
            self._model.update(self._select_update(doutput.y, s_t, is_train=is_train), is_train=is_train)

            # Collecting output
            if self._collect_output:
                collect_output(j, output, alignment, doutput)
            if doutput.a is None:
                alignment = None

        self._model.clean_state()
        
        output = DecodingOutput(output, alignment) if self._collect_output else None
        return (accum_loss / gen_limit), output

    def _classify(self, x_data, gen_limit=50, beam=1, *args, **kwargs):
        # Unpacking
        xp         = self._xp
        batch_size = len(x_data)
        src_len    = len(x_data[0])
        EOL        = self._model._trg_voc.eos_id()
        
        # Perform encoding + Reset state
        self._model.reset_state(x_data, is_train=False, *args, **kwargs)

        output    = np.zeros((batch_size, gen_limit), dtype=np.int32)
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
            beam_search(beam, gen_limit, self._model, x_data, output, alignment, model_update, *args, **kwargs)

        self._model.clean_state()
        return DecodingOutput(output, alignment)

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
def beam_search(beam_size, gen_limit, model, src, output, alignment, model_update, *args, **kwargs):
    queue     = [(0, model.get_state(), 0, DecodingOutput(), None)] # (num_decoded, state, log prob, output, parent)
    beam_size = min(beam_size, len(model._trg_voc))
    EOL       = model._trg_voc.eos_id()
    end_state = None
    # Beam search algorithm
    for loop in range(gen_limit):
        states = []
        reach_end = False
        for i, state in enumerate(queue):
            decoded, dec_state, prob, out, parent = state
            end_state = state
            if i == 0 and out.y == EOL:
                reach_end = True
                break
            
            if loop != 0:
                model.set_state(dec_state)
                model_update(out.y)

            # Conceive the next state
            doutput = model(src, is_train=False, *args, **kwargs)
            dec_state = model.get_state()
            # Take the best 1 (beam search only for single decoding)
            y     = F.softmax(doutput.y).data[0]
            out_a = doutput.a.data[0]
            # Take up several words that makes maximum
            agmx = UF.argmax_index(y, beam_size)
            # Update model state based on those words
            for index in agmx:
                states.append((decoded+1, dec_state, prob + math.log(y[index] + eps), DecodingOutput(index, out_a), state))
        if reach_end:
            break

        states = sorted(states, key=lambda x: x[2], reverse=True)
        queue = states[:beam_size]

    # Collecting output (1-best)
    while end_state is not None:
        decoded, state, prob, out, parent = end_state
        decoded-=1
        if parent is not None:
            output[0][decoded] = out.y
            if out.a is not None:
                for i, x_a in enumerate(out.a):
                    alignment[0][decoded][i] = float(x_a)
            
        # Next state
        end_state = parent

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

