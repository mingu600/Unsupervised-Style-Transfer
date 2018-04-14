## Machine Translation with PyTorch

This repository contains an implementation of a Seq2seq neural network model for machine translation. More details on sequence to sequence machine translation and hyperparameter tuning may be found in [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906).

To train our model, clone the repo and run `main.py`:
```
usage: main.py [-h] [--lr N] [--hs N] [--emb N] [--nlayers N] [--dp N]
               [--unidir] [--attn STR] [--reverse_input] [-v N] [-b N]
               [--epochs N] [--model DIR] [-e] [--visualize] [--predict DIR]
               [--predict_outfile DIR] [--predict_from_input STR]

Machine Translation with Attention

optional arguments:
  -h, --help            show this help message and exit
  --lr N                learning rate, default: 2e-3
  --hs N                size of hidden state, default: 300
  --emb N               embedding size, default: 300
  --nlayers N           number of layers in rnn, default: 2
  --dp N                dropout probability, default: 0.30
  --unidir              use unidirectional encoder, default: bidirectional
  --attn STR            attention: dot-product, additive or none, default:
                        dot-product
  --reverse_input       reverse input to encoder, default: False
  -v N                  vocab size, use 0 for maximum size, default: 0
  -b N                  batch size, default: 64
  --epochs N            number of epochs, default: 50
  --model DIR           path to model, default: None
  -e, --evaluate        only evaluate model, default: False
  --visualize           visualize model attention distribution
  --predict DIR         directory with final input data for predictions,
                        default: None
```

For example, to train with the default parameters, run:

``` python main.py ```

Note: To generate smaller or larger versions of the training dataset for experimenting with the model, you may use the following commands.
 - ```cp .data/iwslt/de-en/train.de-en-full.en .data/iwslt/de-en/train.de-en.en && cp .data/iwslt/de-en/train.de-en-full.de .data/iwslt/de-en/train.de-en.de```
 - ```cp .data/iwslt/de-en/train.de-en-small.en .data/iwslt/de-en/train.de-en.en && cp .data/iwslt/de-en/train.de-en-small.de .data/iwslt/de-en/train.de-en.de```
