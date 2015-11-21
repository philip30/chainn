# chainn
NLP toolkit implementation with neural network. Using [chainer](https://github.com/pfnet/chainer) toolkit. 
You have to install it firstly in order to use this toolkit.
Currently this repository covers some implementations of POS-Tagger, RNN-LM, LSTM-LM, and Encoder-Decoder Model.

# Usage
While some toolkits such as POS-Tagger (rnn-pos.py) and Language model (lm.py) are not supported to output a model (instead, training & testing are conducted in the same run), the decoder is ready to output the model.
To train a model using a default setting you can specify the command bellow:

```python3 train-smt.py --src [SRC_FILE] --trg [TRG_FILE] --model_out [MODEL_OUT]```

To use the model to decode the test file simply run the command bellow:

```python3 --init_model [MODEL_OUT] < [TEST_FILE]```

# Reference
This repository is inspired from [chainer_examples](https://github.com/odashi/chainer_examples) but written in more object oriented fashioned.

# Contact
I am open for any question! contact me at:
* philip[dot]arthur30[at]gmail[dot]com
* philip[dot]arthur[dot]om0[at]is[dot]naist[dot]jp
