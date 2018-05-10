import itertools, os, sys, time
sys.path.append('../')

import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

from validate import validate
from utils import AverageMeter, exclusive_uniform, noisy_sample, distance_constrained_shuffle, Logger

# calculate loss at each individual state
def PGLoss(probs, tokens, rewards):
    N = tokens.size(0)
    C = probs.size(1)
    one_hot = torch.zeros((N, C))
    one_hot.scatter_(1, target.data.view((-1,1)), 1)
    one_hot = one_hot.type(torch.ByteTensor)
    one_hot = Variable(one_hot)
    if prob.is_cuda:
        one_hot = one_hot.cuda()
    loss = torch.masked_select(prob, one_hot)
    loss = loss * reward
    loss =  -torch.sum(loss)
    return loss

# perform specified specified number of rollouts and get reward on expectation
def rollout(y, num, discriminator, model, z, lang_out, out_e, src):
    rewards = []
    seq_len = y.size(1)
    batch_size = y.size(0)

    # loop over all rollouts
    for i in range(num):
        for j in range(1, seq_len):
            data = y[:,:j]
            samples = model.sample(seq_len, batch_size, z, lang_out, out_e, src, y)
            pred = discriminator(samples)
            pred = pred.cpu().data.numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[j-1] += pred

        # TODO: LAST TOKEN ?? also how to deal with different sequence lengths

    rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
    return rewards    

    
def train(train_iter_pos, train_iter_neg, val_iter_pos, val_iter_neg, model, disc_pos, disc_neg, recons_pos, recons_neg,\
          mod_optimizer, disc_pos_optimizer, disc_neg_optimizer, scheduler, POS, NEG, num_epochs, logger=None, num_rollouts=5):

    
    # Iterate through epochs
    bleu_best = -1
    itertrain_pos = iter(train_iter_pos)
    itertrain_neg = iter(train_iter_neg)
    num_batches = max(len(train_iter_pos), len(train_iter_neg))

    d_loss = nn.BCELoss(size_average=False)
    for epoch in range(num_epochs):

        # Validate model with BLEU
        start_time = time.time() # timer
        bleu_val = validate(val_iter_pos, val_iter_neg, model, POS, NEG)
        if bleu_val > bleu_best:
            bleu_best = bleu_val
            logger.save_model(model.state_dict())
            logger.log('New best: {:.3f}'.format(bleu_best))
        val_time = time.time()
        logger.log('Validation time: {:.3f}'.format(val_time - start_time))

        # Step learning rate scheduler
        scheduler.step(bleu_val) # input bleu score

        # Train model
        model.train()
        recon_losses = AverageMeter()
        cross_losses = AverageMeter()
        disc_losses = AverageMeter()
        gen_losses = AverageMeter()

        for i in range(num_batches):
            # Use GPU

            mod_optimizer.zero_grad()

            # POS REINFORCE LOSS
            pos_batch = next(itertrain_pos)
            pos_text = pos_batch.text.cuda() if use_gpu else pos_batch.text

            # get embeddings for our pos text, generate sentences
            out_e, z = model.encoder(pos_text, 0)
            pos_y = model.greedy(pos_text, 0, 1)

            # get rewards based on rollouts, calculate loss
            rewards = rollout(pos_y, num_rollouts, disc_neg, model, z, 1, out_e, pos_text)
            probs = model.decoder_step(pos_y[:,:-1])
            chosen = pos_y[:,1:]
            loss = PGLoss(probs, chosen, rewards)

            # backpropagate
            mod_optimizer.zero_grad()
            loss.backward()
            mod_optimizer.step()
            
            # NEG REINFORCE LOSS
            neg_batch = next(itertrain_neg)
            neg_text = neg_batch.text.cuda() if use_gpu else neg_batch.text

            # get embeddings for our neg text, generate sentences
            out_e, z = model.encoder(neg_text, 1)
            neg_y = model.greedy(neg_text, 1, 0)

            # get rewards based on rollouts, calculate loss
            rewards = rollout(neg_y, num_rollouts, disc_neg, model, z, 0, out_e, neg_text)
            probs = model.decoder_step(neg_y[:,:-1])
            chosen = neg_y[:,1:]
            loss = PGLoss(probs, chosen, rewards)

            # backpropagate
            mod_optimizer.zero_grad()
            loss.backward()
            mod_optimizer.step()

            # Clip gradient norms and step optimizer
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            mod_optimizer.step()
            mod_optimizer.zero_grad()

        # TODO: IMPLEMENT DISCRIMINATOR TRAINING
        for j in range(num_batches):

            disc_optimizer_neg.zero_grad()
            disc_optimizer_pos.zero_grad()

            pos_text = pos_batch.text.cuda() if use_gpu else pos_batch.text
            neg_text = neg_batch.text.cuda() if use_gpu else neg_batch.text

            pos_y = model.greedy(pos_text, 0, 1)
            neg_y = model.greedy(neg_text, 1, 0)

            loss_real_pos = d_loss(disc_pos(pos_text),torch.ones(pos_text.size(1)))
            less_real_neg = d_loss(disc_neg(neg_text),torch.ones(neg_text.size(1)))

            loss_fake_pos = d_loss(disc_pos(neg_y),torch.zeros(neg_y.size(1)))
            loss_fake_neg = d_loss(disc_neg(pos_y),torch.zeros(pos_y.size(1)))

            loss_neg = loss_fake_neg + loss_real_neg
            loss_pos = loss_fake_pos + loss_real_pos

            loss_neg.backward()
            loss_pos.backward()
            disc_optimizer_pos.step()
            disc_optimizer_neg.step()

        
