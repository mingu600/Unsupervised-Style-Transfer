import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
use_gpu = torch.cuda.is_available()

class EncoderLSTM(nn.Module):
    def __init__(self, embedding_s1, embedding_s2, h_dim, num_layers, dropout_p=0.0, bidirectional=True):
        super(EncoderLSTM, self).__init__()
        self.vocab_size_1, self.embedding_size_1 = embedding_s1.size()
        self.vocab_size_2, self.embedding_size_2 = embedding_s2.size()
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional

        # Create word embedding and LSTM
        self.embedding_s1 = nn.Embedding(self.vocab_size_1, self.embedding_size_1)
        self.embedding_s1.weight.data.copy_(embedding_s1)
        self.embedding_s2 = nn.Embedding(self.vocab_size_2, self.embedding_size_2)
        self.embedding_s2.weight.data.copy_(embedding_s2)
        self.lstm = nn.LSTM(self.embedding_size_1, self.h_dim, self.num_layers, dropout=self.dropout_p, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, lang_in, vector_input=False):
        # Embed text
        if lang_in == 0:
            x = self.embedding_s1(x)
        else:
            x = self.embedding_s2(x)
        x = self.dropout(x)

        # Create initial hidden state of zeros: 2-tuple of num_layers x batch size x hidden dim
        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
        init = Variable(torch.zeros(num_layers, x.size(1), self.h_dim), requires_grad=False)
        init = init.cuda() if use_gpu else init
        h0 = (init, init.clone())

        # Pass through LSTM
        out, h = self.lstm(x, h0) # maybe have to pad now?
        return out, h

class Discriminator(nn.Module):
    def __init__(self, latent_dim = 600, hidden_dim = 1024):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(latent_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = F.sigmoid(self.layer3(x))
        return torch.mean(x, dim=0)

class DecoderLSTM(nn.Module):
    def __init__(self, embedding_s1, embedding_s2, h_dim, num_layers, dropout_p=0.0):
        super(DecoderLSTM, self).__init__()
        self.vocab_size_1, self.embedding_size_1 = embedding_s1.size()
        self.vocab_size_2, self.embedding_size_2 = embedding_s2.size()
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.dropout_p = dropout_p

        # Create word embedding, LSTM
        self.embedding_s1 = nn.Embedding(self.vocab_size_1, self.embedding_size_1)
        self.embedding_s1.weight.data.copy_(embedding_s1)
        self.embedding_s2 = nn.Embedding(self.vocab_size_2, self.embedding_size_2)
        self.embedding_s2.weight.data.copy_(embedding_s2)
        self.lstm = nn.LSTM(self.embedding_size_1, self.h_dim, self.num_layers, dropout=self.dropout_p)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x, h0, lang_out, vector_input = False):
        if lang_out == 0:
            x = self.embedding_s1(x)
        else:
            x = self.embedding_s2(x)
        x = self.dropout(x)
        out, h = self.lstm(x, h0)
        return out, h

class Attention(nn.Module):
    def __init__(self, pad_token=1, bidirectional=True, attn_type='dot-product', h_dim=300):
        super(Attention, self).__init__()
        # Check attn type and store variables
        if attn_type not in ['dot-product', 'additive', 'none']:
            raise Exception('Incorrect attention type')
        self.bidirectional = bidirectional
        self.attn_type = attn_type
        self.h_dim = h_dim
        self.pad_token = pad_token

        # Create parameters for additive attention
        if self.attn_type == 'additive':
            self.linear = nn.Linear(2 * self.h_dim, self.h_dim)
            self.tanh = nn.Tanh()
            self.vector = nn.Parameter(torch.zeros(self.h_dim))

    def attention(self, in_e, out_e, out_d):
        '''Produces context and attention distribution'''

        # If no attention, return context of zeros
        if self.attn_type == 'none':
            return out_d.clone() * 0, out_d.clone() * 0

        # Deal with bidirectional encoder, move batches first
        if self.bidirectional: # sum hidden states for both directions
            out_e = out_e.contiguous().view(out_e.size(0), out_e.size(1), 2, -1).sum(2).view(out_e.size(0), out_e.size(1), -1)
        out_e = out_e.transpose(0,1) # b x sl x hd
        out_d = out_d.transpose(0,1) # b x tl x hd

        # Different types of attention
        if self.attn_type == 'dot-product':
            attn = out_e.bmm(out_d.transpose(1,2)) # (b x sl x hd) (b x hd x tl) --> (b x sl x tl)
        elif self.attn_type == 'additive':
            # Resize output tensors for efficient matrix multiplication, then apply additive attention
            bs_sl_tl_hdim = (out_e.size(0), out_e.size(1), out_d.size(1), out_e.size(2))
            out_e_resized = out_e.unsqueeze(2).expand(bs_sl_tl_hdim) # b x sl x tl x hd
            out_d_resized = out_d.unsqueeze(1).expand(bs_sl_tl_hdim) # b x sl x tl x hd
            attn = self.linear(torch.cat((out_e_resized, out_d_resized), dim=3)) # --> b x sl x tl x hd
            attn = self.tanh(attn) @ self.vector # --> b x sl x tl

        # Softmax and reshape
        attn = attn.exp() / attn.exp().sum(dim=1, keepdim=True) # in updated pytorch, make softmax
        attn = attn.transpose(1,2) # --> b x tl x sl

        # Get attention distribution
        context = attn.bmm(out_e) # --> b x tl x hd
        context = context.transpose(0,1) # --> tl x b x hd

        return context, attn

    def forward(self, in_e, out_e, out_d):
        'Produces context vector'
        context, attn = self.attention(in_e, out_e, out_d)
        return context

    def get_visualization(self, in_e, out_e, out_d):
        'Visualization of attention distribution'
        context, attn = self.attention(in_e, out_e, out_d)
        return attn


class Denoise_AE(nn.Module):
    def __init__(self, embedding_e1, embedding_e2, embedding_d1, embedding_d2, h_dim, num_layers, dropout_p, bi, attn_type, tokens_bos_eos_pad_unk=[0,1,2,3], reverse_input=False):
        super(Denoise_AE, self).__init__()
        # Store hyperparameters
        self.h_dim = h_dim
        self.vocab_size_1, self.emb_dim_1 = embedding_e1.size()
        self.vocab_size_2, self.emb_dim_2 = embedding_e2.size()
        self.bos_token = tokens_bos_eos_pad_unk[0]
        self.eos_token = tokens_bos_eos_pad_unk[1]
        self.pad_token = tokens_bos_eos_pad_unk[2]
        self.unk_token = tokens_bos_eos_pad_unk[3]
        self.reverse_input = reverse_input
        # Create encoder, decoder, attention
        self.encoder = EncoderLSTM(embedding_e1, embedding_e2, h_dim, num_layers, dropout_p=dropout_p, bidirectional=bi)
        self.decoder = DecoderLSTM(embedding_d1, embedding_d2, h_dim, num_layers * 2 if bi else num_layers, dropout_p=dropout_p)
        self.attention = Attention(pad_token=self.pad_token, bidirectional=bi, attn_type=attn_type, h_dim=self.h_dim)
        # Create linear layers to combine context and hidden state
        self.linear_com_s1 = nn.Linear(2 * self.h_dim, h_dim)
        self.linear_com_s2 = nn.Linear(2 * self.h_dim, h_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p)
        self.linear_out_s1 = nn.Linear(h_dim, self.vocab_size_1)
        self.linear_out_s2 = nn.Linear(h_dim, self.vocab_size_2)
        # Tie weights of decoder embedding and output
        self.linear_out_s1.weight = self.decoder.embedding_s1.weight
        self.linear_out_s2.weight = self.decoder.embedding_s2.weight

    def forward(self, src, trg, lang_in, lang_out, vector_input = False):
        if use_gpu: src = src.cuda()
        # Reverse src tensor
        if self.reverse_input:
            inv_index = torch.arange(src.size(0)-1, -1, -1).long()
            if use_gpu: inv_index = inv_index.cuda()
            src = src.index_select(0, inv_index)
        # Encode
        out_e, final_e = self.encoder(src, lang_in, vector_input)
        # Decode
        out_d, final_d = self.decoder(trg, final_e, lang_out)
        # Attend
        context = self.attention(src, out_e, out_d)
        out_cat = torch.cat((out_d, context), dim=2)
        # Predict (returns probabilities)
        if lang_out == 0:
            x = self.linear_com_s1(out_cat)
            x = self.dropout(self.tanh(x))
            x = self.linear_out_s1(x)
        else:
            x = self.linear_com_s2(out_cat)
            x = self.dropout(self.tanh(x))
            x = self.linear_out_s2(x)
        return x

    def greedy(self, sentences, lang_in, lang_out, max_len=30, train=False):
        if use_gpu: entences = sentences.cuda()
        num_sent, batch_size = sentences.size()
        outputs_e, states = self.encoder(sentences, lang_in)
        translations = [[self.bos_token] for i in range(batch_size)]
        prev_words = batch_size *[self.bos_token]
        pending = set(range(batch_size))
        output = Variable(torch.zeros(batch_size, self.decoder.h_dim), requires_grad=False)
        first = True
        while len(pending) > 0:
            var = Variable(torch.LongTensor([prev_words]), requires_grad=False).cuda()
            outputs_d, new_state = self.decoder(var, states, lang_out)
            states = new_state
            # Attend
            context = self.attention(sentences, outputs_e, outputs_d)
            out_cat = torch.cat((outputs_d, context), dim=2)
            if lang_out == 0:
                x = self.linear_com_s1(out_cat)
                x = self.dropout(self.tanh(x))
                x = self.linear_out_s1(x)
            else:
                x = self.linear_com_s2(out_cat)
                x = self.dropout(self.tanh(x))
                x = self.linear_out_s2(x)
            x = x.squeeze().clone()
            logprobs = torch.log(x.exp() / x.exp().sum()) # log softmax
            prev_words = logprobs.max(dim=1)[1].squeeze().data.cpu().numpy().tolist()
            for i in pending.copy():
                translations[i].append(prev_words[i])
                if prev_words[i] == self.eos_token:
                    pending.discard(i)
                else:
                    if len(translations[i]) >= max_len:
                        pending.discard(i)
        greedy_result = Variable(torch.t(torch.LongTensor(self.pad(translations))))
        if use_gpu: greedy_result = greedy_result.cuda()
        return greedy_result

    def pad(self, unpadded):
        batch_size = len(unpadded)
        lengths = [len(x) for x in unpadded]
        max_length = max(lengths)
        if max_length == min(lengths):
            return unpadded
        padded = torch.LongTensor(batch_size, max_length).fill_(self.pad_token)
        for i in range(batch_size):
            padded[i][:lengths[i]] = torch.LongTensor(np.array(unpadded[i]))
        return padded

    def predict(self, src, lang_in, lang_out, beam_size=1):
        '''Predict top 1 sentence using beam search. Note that beam_size=1 is greedy search.'''
        beam_outputs = self.beam_search(src, beam_size, lang_in, lang_out, max_len=30) # returns top beam_size options (as list of tuples)
        top1 = beam_outputs[0][1] # a list of word indices (as ints)
        return top1

    def predict_k(self, src, k, lang_in, lang_out, max_len=30, remove_tokens=[]):
        '''Predict top k possibilities for first max_len words.'''
        beam_outputs = self.beam_search(src, k, lang_in, lang_out, max_len=max_len, remove_tokens=remove_tokens) # returns top k options (as list of tuples)
        topk = [option[1] for option in beam_outputs] # list of k lists of word indices (as ints)
        return topk

    def beam_search(self, src, beam_size,lang_in, lang_out, max_len, remove_tokens=[]):
        '''Returns top beam_size sentences using beam search. Works only when src has batch size 1.'''
        if use_gpu: src = src.cuda()
        # Reverse src tensor
        if self.reverse_input:
            inv_index = torch.arange(src.size(0)-1, -1, -1).long()
            if use_gpu: inv_index = inv_index.cuda()
            src = src.index_select(0, inv_index)
        # Encode
        outputs_e, states = self.encoder(src, lang_in)
        # Start with '<s>'
        init_lprob = -1e10
        init_sent = [self.bos_token]
        best_options = [(init_lprob, init_sent, states)] # beam
        # Beam search
        k = beam_size # store best k options
        for length in range(max_len): # maximum target length
            options = [] # candidates
            for lprob, sentence, current_state in best_options:
                # Prepare last word
                last_word = sentence[-1]
                if last_word != self.eos_token:
                    last_word_input = Variable(torch.LongTensor([last_word]), volatile=True).view(1,1)
                    if use_gpu: last_word_input = last_word_input.cuda()
                    # Decode
                    outputs_d, new_state = self.decoder(last_word_input, current_state, lang_out)
                    # Attend
                    context = self.attention(src, outputs_e, outputs_d)
                    out_cat = torch.cat((outputs_d, context), dim=2)

                    if lang_out == 0:
                        x = self.linear_com_s1(out_cat)
                        x = self.dropout(self.tanh(x))
                        x = self.linear_out_s1(x)
                    else:
                        x = self.linear_com_s2(out_cat)
                        x = self.dropout(self.tanh(x))
                        x = self.linear_out_s2(x)

                    x = x.squeeze().data.clone()
                    # Block predictions of tokens in remove_tokens
                    for t in remove_tokens: x[t] = -10e10
                    lprobs = torch.log(x.exp() / x.exp().sum()) # log softmax
                    # Add top k candidates to options list for next word
                    for index in torch.topk(lprobs, k)[1]:
                        option = (float(lprobs[index]) + lprob, sentence + [index], new_state)
                        options.append(option)
                else: # keep sentences ending in '</s>' as candidates
                    options.append((lprob, sentence, current_state))
            options.sort(key = lambda x: x[0], reverse=True) # sort by lprob
            best_options = options[:k] # place top candidates in beam
        best_options.sort(key = lambda x: x[0], reverse=True)
        return best_options

    def get_attn_dist(self, src, trg, lang_in, lang_out):
        '''Runs forward pass, also returns attention distribution'''
        if use_gpu: src = src.cuda()
        # Reverse src tensor
        if self.reverse_input:
            inv_index = torch.arange(src.size(0)-1, -1, -1).long()
            if use_gpu: inv_index = inv_index.cuda()
            src = src.index_select(0, inv_index)
        # Encode, Decode, Attend
        out_e, final_e = self.encoder(src, lang_in)
        out_d, final_d = self.decoder(trg, final_e, lang_out)
        context = self.attention(src, out_e, out_d)
        out_cat = torch.cat((out_d, context), dim=2)
        # Predict
        if lang_out == 0:
            x = self.linear_com_s1(out_cat)
            x = self.dropout(self.tanh(x))
            x = self.linear_out_s1(x)
        else:
            x = self.linear_com_s2(out_cat)
            x = self.dropout(self.tanh(x))
            x = self.linear_out_s2(x)
        # Visualize attention distribution
        attn_dist = self.attention.get_visualization(src, out_e, out_d)
        return x, attn_dist
