import numpy as np
from nltk import sent_tokenize
import nltk.data
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import torchtext
from torchtext.vocab import Vectors, GloVe
from torchtext import data
import spacy
import torch.nn as nn
import time
from utils import Logger, AverageMeter
from torch.autograd import Variable
import torch



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
            sentences += sentence_splitter.tokenize(txt)
            for sent in sentences[2:]:
                text = []
                text += text_field.preprocess(sent)
                text += ['</s>']
                if 3 <= len(text) <= 29:
                    examples.append(data.Example.fromlist([text + ['<pad>'] *(29- len(text))], fields))

        else:
            examples = examples
        super(EncoderDataset, self).__init__(
            examples, fields, **kwargs)


def preprocess(vocab_size, batchsize, preprocessed = True, max_sent_len=20):
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

class EncoderRNN(nn.Module):
    def __init__(self, embedding, h_dim, num_layers, use_gpu, dropout_p=0.0, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.vocab_size, self.embedding_size = embedding.size()
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        self.use_gpu = use_gpu

        # Create word embedding and LSTM
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(embedding)
        # self.embedding.weight.requires_grad=False
        self.lstm = nn.LSTM(self.embedding_size, self.h_dim, self.num_layers, dropout=self.dropout_p, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p)
        nn.init.xavier_normal(self.lstm.weight_ih_l0)
        nn.init.xavier_normal(self.lstm.weight_hh_l0)
        nn.init.constant(self.lstm.bias_ih_l0, 1)
        nn.init.constant(self.lstm.bias_hh_l0, 1)

    def forward(self, x):
        # Embed text
        x = self.embedding(x)
        x = self.dropout(x)
        out, h = self.lstm(x)
        return out, h

def score_function(encodings1):
    prod = torch.exp(torch.mm(nn.functional.normalize(encodings1, dim=1), torch.t(nn.functional.normalize(encodings1, dim=1))))
    normalized = prod-Variable(np.e * torch.eye(prod.size()[0]), requires_grad = False).cuda()
    normalized = nn.functional.normalize(normalized, p=1, dim=1)
    probs = torch.log(torch.diag(normalized, diagonal=1)[1:]) + torch.log(torch.diag(normalized, -1)[:-1])
    return -1 * torch.sum(probs) / len(probs)


def train(train_iter, val_iter, encoder1, enc_optimizer1 ,  TEXT,  use_gpu, num_epochs,logger=None):
    # Iterate through epochs
    num_batches = len(train_iter)
    print(num_batches)
    for epoch in range(num_epochs):
        start_time = time.time()
        # Train model
        encoder1.train()
        #encoder2.train()
        losses = AverageMeter()
        i = 0
        for batch in train_iter:
            enc_optimizer1.zero_grad()
            #enc_optimizer2.zero_grad()
            # POS RECONSTRUCTION LOSS
            text = batch.text[:-1].cuda() if use_gpu else batch.text
            output1, final_h1 = encoder1(text)
            #output2, final_h2 = encoder2(text)
            #loss = score_function(torch.cat((final_h1[0][0], final_h1[1][0]), 1), torch.cat((final_h2[0][0], final_h2[1][0]), 1))
            #loss = score_function(torch.cat((final_h1[0], final_h1[1]), 1), torch.cat((final_h2[0], final_h2[1]), 1))
            loss = score_function(torch.cat((final_h1[0][0], final_h1[1][0]), 1))
            loss.backward()
            losses.update(loss.data[0])
            torch.nn.utils.clip_grad_norm(encoder1.parameters(), 10)
            #torch.nn.utils.clip_grad_norm(encoder2.parameters(), 5)
            enc_optimizer1.step()
            #enc_optimizer2.step()


            # # Log within epoch
            if i % 100 == 0:
                print('Epoch [{}/{}]\t Batch [{}/{}]\t Loss: {:f}'.format(epoch+1, num_epochs, i, num_batches,losses.avg))
            i += 1
        # Log after each epoch
        print('Epoch [{}/{}] Complete. \t Loss: {:f}'.format(epoch+1, num_epochs, losses.avg))
        print('Training time: {:f}'.format(time.time() - start_time))
        #torch.save(encoder1.state_dict(), 'encoder_models/encoder_1_epoch_' + str(epoch) + '.pt')
        #torch.save(encoder2.state_dict(), 'encoder_models/encoder_2_epoch_' + str(epoch) + '.pt')
        torch.save(encoder1.state_dict(), 'encoder_models/encoder_1_300_epoch_' + str(60 + epoch) + '.pt')

def main():
    use_gpu = True
    print('Loading text...')
    time_data = time.time()
    TEXT, train_iter, val_iter = preprocess(50000, 48, False)
    print('Finished loading data. |Vocab| = {}. Time: {:.2f}.'.format(len(TEXT.vocab), time.time() - time_data))
    #encoder = EncoderRNN((torch.rand(len(TEXT.vocab), 300) - 0.5) / 5.0, 300, num_layers = 1, use_gpu = use_gpu, dropout_p = 0, bidirectional=False)
    #encoder1 = EncoderRNN((torch.rand(len(TEXT.vocab), 300) - 0.5) / 5.0, 500, num_layers = 1, use_gpu = use_gpu, dropout_p = 0, bidirectional=False)
    encoder1 = EncoderRNN(TEXT.vocab.vectors, 300, num_layers = 1, use_gpu = use_gpu, dropout_p = 0.3, bidirectional=True)
    #encoder2 = EncoderRNN(TEXT.vocab.vectors, 500, num_layers = 1, use_gpu = use_gpu, dropout_p = 0.0, bidirectional=True)
    if use_gpu:
        encoder1.cuda()
        #encoder2.cuda()
    # Create loss function and optimizer
    encoder1.load_state_dict(torch.load('encoder_models/encoder_1_300_epoch_60.pt'))
    enc_params1 = filter(lambda p: p.requires_grad, encoder1.parameters())
    enc_optimizer1 = torch.optim.Adam(enc_params1, lr=1e-4)
    #enc_params2 = filter(lambda p: p.requires_grad, encoder2.parameters())
    #enc_optimizer2 = torch.optim.Adam(enc_params2, lr=5e-4)
    train(train_iter, val_iter, encoder1,enc_optimizer1, TEXT, use_gpu,num_epochs = 300)

if __name__ == '__main__':
    main()
