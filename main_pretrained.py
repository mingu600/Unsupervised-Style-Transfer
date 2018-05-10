import argparse, os, datetime, time, sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext
from torchtext import data
from torchtext.vocab import Vectors, GloVe
from train import train, train_2
from encoder_train import EncoderDataset
import spacy
from validate import validate
from utils import Logger, AverageMeter
from load_data import preprocess, load_embeddings
from models import Denoise_AE, Discriminator, EncoderLSTM, EncoderLSTM_v2, Denoise_AE_v2

parser = argparse.ArgumentParser(description='Unsupervised Machine Translation with Attention')
parser.add_argument('--lr', default=5e-4, type=float, metavar='N', help='learning rate, default: 2e-3')
parser.add_argument('--hs', default=300, type=int, metavar='N', help='size of hidden state, default: 300')
parser.add_argument('--emb', default=300, type=int, metavar='N', help='embedding size, default: 300')
parser.add_argument('--nlayers', default=1, type=int, metavar='N', help='number of layers in rnn, default: 2')
parser.add_argument('--dp', default=0.0, type=float, metavar='N', help='dropout probability, default: 0.30')
parser.add_argument('--unidir', dest='bi', action='store_false', help='use unidirectional encoder, default: bidirectional')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of epochs, default: 50')
parser.add_argument('--attn', default='dot-product', type=str, metavar='STR', help='attention: dot-product, additive or none, default: dot-product ')
parser.add_argument('--reverse_input', dest='reverse_input', action='store_true', help='reverse input to encoder, default: False')
parser.add_argument('-v', default=0, type=int, metavar='N', help='vocab size, use 0 for maximum size, default: 0')
parser.add_argument('-b', default=64, type=int, metavar='N', help='batch size, default: 64')
parser.add_argument('--model', metavar='DIR', default=None, help='path to model, default: None')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='only evaluate model, default: False')
parser.add_argument('--visualize', dest='visualize', action='store_true', help='visualize model attention distribution')
parser.set_defaults(evaluate=False, bi=True, reverse_input=False, visualize=False)

def preprocess_2(vocab_size, batchsize, preprocessed = True, max_sent_len=20):
    '''Loads data from text files into iterators'''

    # Load text tokenizers
    spacy_en = spacy.load('en')

    def tokenize(text, lang='en'):
        if lang is 'en':
            return [tok.text for tok in spacy_en.tokenizer(text)]
        else:
            raise Exception('Invalid language')

    # Add beginning-of-sentence and end-of-sentence tokens
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    TEXT = data.Field(tokenize=tokenize, init_token=BOS_WORD, eos_token=EOS_WORD)
    if preprocessed:
        train = EncoderDataset(path = 'encoder_data/combined_shakespeare.txt',text_field = TEXT, examples = torch.load('shake_train.pt'))
        val = EncoderDataset(path='encoder_data/combined_shakespeare.txt', text_field = TEXT, examples = torch.load('shake_val.pt'))
    else:
        train = EncoderDataset(path='encoder_data/combined_shakespeare.txt', text_field = TEXT)
        val = EncoderDataset(path='encoder_data/combined_shakespeare.txt', text_field = TEXT)
        torch.save(train.examples, 'shake_train.pt')
        torch.save(val.examples, 'shake_val.pt')
    if vocab_size > 0:
        TEXT.build_vocab(train.text, min_freq=4, max_size=vocab_size)
    else:
        TEXT.build_vocab(train.text, min_freq=4)
    # Create iterators to process text in batches of approx. the same length
    train_iter = data.Iterator(train, batch_size=batchsize, device=-1, repeat=False, shuffle=False, sort=False)
    val_iter = data.Iterator(val, batch_size=batchsize, device=-1, repeat=False, shuffle=False, sort=False)
#     url = 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
    # print('Loading vectors...')
    # TEXT.vocab.load_vectors(vectors=GloVe(name='42B', dim=300))
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
    return TEXT, train_iter, val_iter

def main():
    global args
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()


    TEXT, _, _ = preprocess_2(args.v, args.b, False)
    print(len(TEXT.vocab))
    # Load and process data
    print('Loading dataset')
    time_data = time.time()
    POS, NEG, train_iter_pos, train_iter_neg, val_iter_pos, val_iter_neg = preprocess(args.v, args.b)
    print('Loaded data. |POS| = {}, |NEG| = {}. Time: {:.2f}.'.format(len(POS.vocab), len(NEG.vocab), time.time() - time_data))

    embedding_e1 = (torch.rand(len(POS.vocab), args.emb) - 0.5) * 2
    embedding_e2 = (torch.rand(len(NEG.vocab), args.emb) - 0.5) * 2
    embedding_d1 = (torch.rand(len(POS.vocab), args.emb) - 0.5) * 2
    embedding_d2 = (torch.rand(len(NEG.vocab), args.emb) - 0.5) * 2
    print('Initialized embedding vectors')

    # Create model
    tokens = [NEG.vocab.stoi[x] for x in ['<s>', '</s>', '<pad>', '<unk>']]
    model_enc = EncoderLSTM_v2(torch.rand(5147, args.emb), args.hs, args.nlayers, dropout_p=args.dp, bidirectional=args.bi)
    model_enc.load_state_dict(torch.load('encoder_models/encoder_1_300_epoch_60.pt'))
    model = Denoise_AE_v2(torch.rand(5147, args.emb), embedding_d1, embedding_d2,args.hs, args.nlayers, args.dp, args.bi, args.attn, tokens_bos_eos_pad_unk=tokens, reverse_input=args.reverse_input)
    model.encoder = model_enc
    #discrim = Discriminator(args.hs * 2 * args.nlayers if args.bi == True else args.hs * args.nlayers, 1024)
    # Load pretrained model
    if args.model is not None and os.path.isfile(args.model):
        model.load_state_dict(torch.load(args.model))
        print('Loaded pretrained model.')
    model = model.cuda() if use_gpu else model
    #discrim = discrim.cuda() if use_gpu else discrim

    # Create weight to mask padding tokens for loss function
    weight_1 = torch.ones(len(POS.vocab))
    weight_1[POS.vocab.stoi['<pad>']] = 0
    weight_1 = weight_1.cuda() if use_gpu else weight_1

    weight_2 = torch.ones(len(NEG.vocab))
    weight_2[NEG.vocab.stoi['<pad>']] = 0
    weight_2 = weight_2.cuda() if use_gpu else weight_2

    # Create loss function and optimizer
    recons_pos = nn.CrossEntropyLoss(weight=weight_1)
    recons_neg = nn.CrossEntropyLoss(weight=weight_2)
    #d_loss = nn.BCELoss(size_average=False)

    enc_params = filter(lambda p: p.requires_grad, model.encoder.parameters())
    for p in enc_params:
        p.requires_grad = False
    mod_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    #dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discrim.parameters()), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mod_optimizer, 'max', patience=30, factor=0.25, verbose=True, cooldown=6)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,13,16,19], gamma=0.5)

    # Create directory for logs, create logger, log hyperparameters
    path = os.path.join('saves', datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    os.makedirs(path, exist_ok=True)
    logger = Logger(path)
    logger.log('COMMAND ' + ' '.join(sys.argv), stdout=False)
    logger.log('ARGS: {}\nOPTIMIZER: {}\nLEARNING RATE: {}\nSCHEDULER: {}\nMODEL: {}\n'.format(args, mod_optimizer, args.lr, vars(scheduler), model), stdout=False)

    # Train, validate, or predict
    start_time = time.time()
    if args.evaluate:
        validate_2(val_iter, model, criterion, SRC, TRG, logger)
    else:
        train_2(train_iter_pos, train_iter_neg, val_iter_pos, val_iter_neg,\
              model,recons_pos, recons_neg ,mod_optimizer, scheduler, POS, NEG, TEXT, args.epochs, logger)
    logger.log('Finished in {}'.format(time.time() - start_time))
    return

if __name__ == '__main__':
    print(' '.join(sys.argv))
    main()
