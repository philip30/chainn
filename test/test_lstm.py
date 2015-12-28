from chainn.test import TestRNN
from chainn.model import LSTMRNN

class TestLSTM(TestRNN):
    def __init__(self, *args, **kwargs):
        super(TestLSTM, self).__init__(*args, **kwargs)
        self.Model = LSTMRNN

