## Unsupervised Style Transfer with PyTorch

This repository contains our personal implementation of an unsupervised neural machine translation model, largely inspired by the following two papers:

Lample, G., Denoyer, L., & Ranzato, M. (2017). [Unsupervised Machine Translation Using Monolingual Corpora Only](https://arxiv.org/pdf/1711.00043.pdf).

Shen, T., Lei, T., Barzilay, R., & Jaakkola, T. (2017). [Style Transfer from Non-Parallel Text by Cross-Alignment](https://arxiv.org/pdf/1705.09655.pdf).

We've implemented much of the key parts of previous unsupervised machine translation model, and have worked on various different improvements.

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

``` python3 main.py ```
