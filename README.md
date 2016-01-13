# chainn
by: Philip Arthur

NLP toolkit implementation with (recursive) neural network. Using [chainer](https://github.com/pfnet/chainer) (v1.5+) toolkit.
You have to install it firstly in order to use this toolkit.
Currently this repository covers implementations of:

 1. POS-Tagger (RNN / LSTM): ```pos.py```
 2. Language Model (RNN / LSTM): ```lm.py```
 3. Neural Machine Translation (Encoder-Decoder, Attentional Model): ```nmt.py```.

*NOTE:*
By default the software will use GPU to do the computation. If you don't have any GPU installed, please specify ``--use_cpu`` in both training and testing.

# Usage
Each of the implementation has their own manual:

 1. POS-tagger
 2. Language Model
 3. NMT

But we provide a general overview of how do you train and test the model using this toolkit.
## Training
For each implementation, you can train the model by running the ```train-i.py``` where ```i``` could be ```pos```, ```lm```, or ```nmt```. For example, to train a pos-tagger with 2 layer of 50 hidden nodes lstm, you can use this command:

```python3 train-pos.py --model lstm --hidden 50 --depth 2 --model_out [model_out] < [train_data]```

The model will be saved in [model\_out] directory, and you have to provide a training data [train\_data].

## Testing
To do testing, you can use the trained model ```[model_out]``` by specifying it at ```--init_model``` options:

```python3 pos.py --init_model [model_out] < [test_data]```

and the software will produce a pos tag for each input.


# Contact
I am open for any question! Or if you found and bug, please contact me at:

* philip[dot]arthur30[at]gmail[dot]com
* philip[dot]arthur[dot]om0[at]is[dot]naist[dot]jp

