import sys
import os
import numpy as np
import chainn.util.functions as UF

from chainer import serializers
from chainn import Vocabulary

class ModelSerializer:
    def __init__(self, directory, compress=False):
        self.directory = directory

    def __enter__(self):
        return self

    def __exit(self):
        # TODO implement compress?
        pass
        
    def save(self, classifier):
        self._init_dir()
        self._write_classifier(classifier)
        self._write_model(classifier._model)
    
    def load(self, classifier, all_models, xp):
        if not os.path.exists(self.directory):
            raise Exception("Could not find directory:", self.directory)
        
        # reading in spefic model name
        with open(os.path.join(self.directory, "model.name")) as model_file:
            model_name    = model_file.readline().strip()
            Model         = UF.select_model(model_name, all_models)
        
        # reading in opt state
        if classifier._opt is not None:
            serializers.load_npz(os.path.join(self.directory, "model.opt"), classifier._opt)
        
        # reading in training state
        with open(os.path.join(self.directory, "model.state")) as state_file:
            training_state = self._read_specification(state_file)
            classifier.set_specification(training_state)

        classifier._model = self._read_model(Model, xp) 

    ## classifier_state
    def _write_classifier(self, classifier):
        with open(os.path.join(self.directory, "model.name"), "w") as model_file:
            model_file.write(classifier._model.__class__.name + "\n")

        # Saving optimizer state
        serializers.save_npz(os.path.join(self.directory, "model.opt"), classifier._opt)
      
        # Saving classifier specification
        with open(os.path.join(self.directory, "model.state"), "w") as state_file:
            self._write_specification(classifier, state_file)
   
    ## model_state
    def _write_model(self, model):
        directory = self.directory
        # Saving model specification
        with open(os.path.join(directory, "model.spec"), "w") as spec_file:
            self._write_specification(model, spec_file)
        with open(os.path.join(directory, "model.src_vocab"), "w") as src_voc_file:
            self._write_vocabulary(model._src_voc, src_voc_file)
        with open(os.path.join(directory, "model.trg_vocab"), "w") as trg_voc_file:
            self._write_vocabulary(model._trg_voc, trg_voc_file)

        serializers.save_npz(os.path.join(directory, "model.weight"), model)

    def _write_vocabulary(self, vocab, fp):
        self.__write_attribute(fp, "length", len(vocab))
        for word, index in sorted(vocab._data.items(), key=lambda x:x[0]):
            fp.write(str(word) + "\t" + str(index) + "\n")

    def _write_specification(self, model, fp):
        for spec_name, spec_val in sorted(model.get_specification().items()):
            self.__write_attribute(fp, spec_name, spec_val)

    def _read_specification(self, spec_file):
        ret = lambda: None
        for line in spec_file:
            spec_name, spec_value = self.__read_attribute(line)
            setattr(ret, spec_name, spec_value)
        return ret

    def _read_vocabulary(self, fp):
        _, size = self.__read_attribute(fp.readline())
        ret = Vocabulary(unk=False,eos=False)
        for i in range(size):
            word, index = fp.readline().strip().split("\t")
            ret[word] = int(index)
        return ret
    
    def _read_model(self, Model, xp=np):
        # reading in vocabularies
        with open(os.path.join(self.directory, "model.src_vocab")) as src_vocab_fp:
            src_voc   = self._read_vocabulary(src_vocab_fp)
        with open(os.path.join(self.directory, "model.trg_vocab")) as trg_vocab_fp:
            trg_voc   = self._read_vocabulary(trg_vocab_fp)

        # reading in model
        with open(os.path.join(self.directory, "model.spec")) as spec_file:
            model_spec = self._read_specification(spec_file)
        
        return Model(src_voc, trg_voc, model_spec, xp=xp)

    ## Helper Function
    def _init_dir(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def __write_attribute(self, fp, name, value):
        if type(value) == float:
            fp.write(name + "\tfloat\t" + "%.30f" % (value) + "\n")
        elif type(value) == int:
            fp.write(name + "\tint\t" + "%d" % (value) + "\n")
        elif type(value) == str:
            fp.write(name + "\tstr\t" + value + "\n")
        elif type(value) == bool:
            fp.write(name + "\tbool\t" + str(value) + "\n")
        else:
            raise ValueError("Undefined attribute type:" + str(type(value)))

    def __read_attribute(self, line):
        name, typ, val = line.strip().split("\t")
        if typ == "float": val = float(val)
        elif typ == "int": val = int(val)
        elif typ == "bool": val = val == "True"
        elif typ == "str": pass
        else: raise ValueError("Unknown type:", typ)
        return name, val

