import random

class Batch(object):
    def __init__(self, batch_id, data):
        self.id   = batch_id
        self.data = data

class IdentityTransform:
    def transform(self, x):
        return x

class BatchState:
    def __init__(self):
        self.id      = 0
        self.data    = []

class BatchManager(object):
    def __init__(self, data_transformer = IdentityTransform()):
        # public
        self.indexes         = []
        # private
        self.index_map       = {}
        self.transformer     = data_transformer

    # stream  : data stream
    # n_items : number of items in batch
    def load(self, stream, n_items):
        assert(n_items >= 1)
        batch_state = BatchState()
        
        def new_batch():
            batch = Batch(\
                    batch_state.id,\
                    self.transformer.transform(batch_state.data))
            self.indexes.append(batch_state.id)
            self.index_map[batch_state.id] = batch
            batch_state.data = []
            batch_state.id  += 1
       
        # Load data from stream
        for i, data in enumerate(stream):
            batch_state.data.append(data)
            if len(batch_state.data) == n_items:
                new_batch()
        
        if len(batch_state.data) != 0:
            new_batch()
    
    def arrange(self, indexes):
        self.indexes = indexes

    def shuffle(self):
        random.shuffle(self.indexes)
    
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        return self.index_map[self.indexes[index]]

    def __iter__(self):
        for index in self.indexes:
            yield self.index_map[index]
