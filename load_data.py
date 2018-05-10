import itertools, os, io
import numpy as np
import spacy

import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe
use_gpu = torch.cuda.is_available()


class EncoderDataset(data.Dataset):
    """Defines a dataset for Encoder Style Transfer."""

    def __init__(self, path, text_field, examples=[],
                 encoding='utf-8', **kwargs):
        """Create a EncoderDataset given a path and a field.
        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        nltk.download('punkt')
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'i.e', 'ltd', 'jr', 'sr', 'co', 'st', 'ms', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sept', 'nov', 'dec'])
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        fields = [('text', text_field)]
        if len(examples) == 0:
            examples = []
            sentences = []
            fp = open(path)
            txt = fp.read()
            txt = txt.replace('?"', '? "').replace('!"', '! "').replace('."', '. "').replace("?'", "? '").replace("!'", "! '").replace(".'", ". '").replace('\n', ' ')
            sentences += sentence_splitter.tokenize(txt.lower())
            for sent in sentences[2:]:
                text = []
                text += text_field.preprocess(sent)
                text += ['</s>']
                if 3 <= len(text) <= 19:
                    examples.append(data.Example.fromlist([text + ['<pad>'] *(19- len(text))], fields))

        else:
            examples = examples
        super(EncoderDataset, self).__init__(
            examples, fields, **kwargs)


class SentimentDataset(data.Dataset):
    """Defines a dataset for Sentiment Transfer."""

    def __init__(self, path, text_field,
                 encoding='utf-8', **kwargs):
        """Create a SentimentDataset given a path and a field.
        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field)]
        examples = []
        with io.open(path, encoding=encoding) as f:
            for line in f:
                text = []
                text += text_field.preprocess(line.lower())
                if len(text) <= 20 and len(text) >= 3:
                    examples.append(data.Example.fromlist([text], fields))
        super(SentimentDataset, self).__init__(
            examples, fields, **kwargs)


# def preprocess_2(vocab_size, batchsize, max_sent_len=20):
#     '''Loads data from text files into iterators'''
#
#     # Load text tokenizers
#     spacy_en = spacy.load('en')
#
#     def tokenize(text, lang='en'):
#         if lang is 'en':
#             return [tok.text for tok in spacy_en.tokenizer(text)]
#         else:
#             raise Exception('Invalid language')
#
#     # Add beginning-of-sentence and end-of-sentence tokens
#     BOS_WORD = '<s>'
#     EOS_WORD = '</s>'
#     TEXT = data.Field(tokenize=tokenize, init_token=BOS_WORD, eos_token=EOS_WORD)
#     train = EncoderDataset(path = 'encoder_data/combined_cbt_news.txt',text_field = TEXT, examples = torch.load('cbt_train.pt'))
#     val = EncoderDataset(path='encoder_data/cbt_valid.txt', text_field = TEXT, examples = torch.load('cbt_val.pt'))
#
#     if vocab_size > 0:
#         TEXT.build_vocab(train.text, min_freq=5, max_size=vocab_size)
#     else:
#         TEXT.build_vocab(train.text, min_freq=5)
#
#     train_pos = SentimentDataset(path='data/sentiment_train_pos.txt', text_field = TEXT)
#     train_neg = SentimentDataset(path='data/sentiment_train_neg.txt', text_field = TEXT)
#     val_pos = SentimentDataset(path='data/sentiment_val_pos.txt', text_field = TEXT)
#     val_neg = SentimentDataset(path= 'data/sentiment_val_neg.txt', text_field = TEXT)
#
#     # Create iterators to process text in batches of approx. the same length
#     train_iter_pos = data.BucketIterator(train_pos, batch_size=batchsize, device=-1, repeat=True, sort_key=lambda x: len(x.text))
#     train_iter_neg = data.BucketIterator(train_neg, batch_size=batchsize, device=-1, repeat=True, sort_key=lambda x: len(x.text))
#     val_iter_pos = data.BucketIterator(val_pos, batch_size=1, device=-1, repeat=True, sort_key=lambda x: len(x.text))
#     val_iter_neg = data.BucketIterator(val_neg, batch_size=1, device=-1, repeat=True, sort_key=lambda x: len(x.text))
#
#     return TEXT, TEXT, train_iter_pos, train_iter_neg, val_iter_pos, val_iter_neg

def preprocess(vocab_size, batchsize, max_sent_len=20):
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
    POS = data.Field(tokenize=tokenize, init_token=BOS_WORD, eos_token=EOS_WORD)
    NEG = data.Field(tokenize=tokenize, init_token=BOS_WORD, eos_token=EOS_WORD)
    train_pos = SentimentDataset(path='data/shakespeare_original.txt', text_field = POS)
    train_neg = SentimentDataset(path='data/shakespeare_modern.txt', text_field = NEG)
    val_pos = SentimentDataset(path='data/shakespeare_original.txt', text_field = POS)
    val_neg = SentimentDataset(path= 'data/shakespeare_modern.txt', text_field = NEG)
    # train_pos = SentimentDataset(path='data/sentiment_train_pos.txt', text_field = POS)
    # train_neg = SentimentDataset(path='data/sentiment_train_neg.txt', text_field = NEG)
    # val_pos = SentimentDataset(path='data/sentiment_val_pos.txt', text_field = POS)
    # val_neg = SentimentDataset(path= 'data/sentiment_val_neg.txt', text_field = NEG)
    if vocab_size > 0:
        POS.build_vocab(train_pos.text, min_freq=3, max_size=vocab_size)
        NEG.build_vocab(train_neg.text, min_freq=3, max_size=vocab_size)
    else:
        POS.build_vocab(train_pos.text, min_freq=3)
        NEG.build_vocab(train_neg.text, min_freq=3)

    # Create iterators to process text in batches of approx. the same length
    train_iter_pos = data.BucketIterator(train_pos, batch_size=batchsize, device=-1, repeat=True, sort_key=lambda x: len(x.text))
    train_iter_neg = data.BucketIterator(train_neg, batch_size=batchsize, device=-1, repeat=True, sort_key=lambda x: len(x.text))
    val_iter_pos = data.BucketIterator(val_pos, batch_size=1, device=-1, repeat=True, sort_key=lambda x: len(x.text))
    val_iter_neg = data.BucketIterator(val_neg, batch_size=1, device=-1, repeat=True, sort_key=lambda x: len(x.text))

    return POS, NEG, train_iter_pos, train_iter_neg, val_iter_pos, val_iter_neg


def load_embeddings(np_emb_e1_file, np_emb_e2_file, np_emb_d1_file, np_emb_d2_file):
    'Load style embeddings from numpy files'
    if os.path.isfile(np_emb_e1_file) and os.path.isfile(np_emb_e2_file) and os.path.isfile(np_emb_d1_file) and os.path.isfile(np_emb_d2_file):
        emb_tr_e1 = torch.from_numpy(np.load(np_emb_e1_file))
        emb_tr_e2 = torch.from_numpy(np.load(np_emb_e2_file))
        emb_tr_d1 = torch.from_numpy(np.load(np_emb_d1_file))
        emb_tr_d2 = torch.from_numpy(np.load(np_emb_d2_file))
    else:
        raise Exception('Vectors not available to load from numpy file')
    return emb_tr_e1, emb_tr_e2, emb_tr_d1, emb_tr_d2
