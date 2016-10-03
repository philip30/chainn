import numpy as np
import chainn.global.xp as xp

from chainn.util import Vocabulary

class NMT_BasicTrainDataTransformer:
    def __init__(self, corpus_analyzer):
        self.corpus_anaylzer = corpus_analyzer

    def transform(self, batched_data):
        srcs = []
        trgs = []
        for src, trg in batched_data:
            srcs.append(src.strip().split() + [EOS])
            trgs.append(trg.strip().split() + [EOS])
        
        max_src = 

