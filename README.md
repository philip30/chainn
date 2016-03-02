# chainn
by: Philip Arthur

NLP toolkit implementation with (recursive) neural network. Using [chainer](https://github.com/pfnet/chainer) (v1.5+) toolkit.
You have to install it firstly in order to use this toolkit.
After installing chainer, please add the folder containing the folder of this project to your PYTHONPATH environment variable.

Currently this repository covers implementations of:

 1. POS-Tagger (RNN / LSTM): ```pos.py```
 2. Language Model (RNN / LSTM): ```lm.py```
 3. Neural Machine Translation (Encoder-Decoder, Attentional Model): ```nmt.py```.

*NOTE:*
By default the software will use CPU to do the computation. In order to use GPU you can specify ```--gpu 0``` for using your first GPU machine, or increase the number to select another.

# Usage
Each of the implementation has their own manual:

 1. POS-tagger
 2. Language Model
 3. NMT

We provide a general overview for training and testing using this toolkit.

## Training POS + LM
For each implementation, you can train the model by running the ```train-i.py``` where ```i``` could be ```pos```, ```lm```. For example, to train a pos-tagger with 2 layer of 50 hidden nodes lstm, you can use this command:

```python3 train-pos.py --model lstm --hidden 50 --depth 2 --model_out [model_out] < [train_data]```

The model will be saved in [model\_out] directory, and you have to provide a training data [train\_data].

## Training NMT
For training NMT model, it is basically the same as training ```pos``` and ```lm``` model but it just has slightly different option:

```python3 train-nmt.py --model attn --hidden 256 --depth 2 --src [SRC_FILE] --trg [TRG_FILE] --model_out [model_out]```

The above command will train attentional neural translation model with 256 hidden node. Currently it supports ```attn``` for attentional model and ```encdec``` for encdec model.

## Testing
To do testing, you can use the trained model ```[model_out]``` by specifying it at ```--init_model``` options:

```python3 pos.py --init_model [model_out] < [test_data]```

and the software will produce a pos tag for each input.

## Other (stable) options

 1. ```--epoch``` to specify how many epoch.
 2. ```--verbose``` to show verbosity during training / testing.
 3. ```--embed``` specify the size of embedding layer.

The other options you find in the program is experimental.

## Note 

 1. By default "Adam" will be used for optimizer.
 
# Contact
I am open for any question! Or if you found and bug, please contact me at:

* philip[dot]arthur30[at]gmail[dot]com
* philip[dot]arthur[dot]om0[at]is[dot]naist[dot]jp

