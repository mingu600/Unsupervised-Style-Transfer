import itertools, os, sys, time
sys.path.append('../')

import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

from validate import validate
from utils import AverageMeter, exclusive_uniform, noisy_sample, distance_constrained_shuffle, Logger



def train(train_iter_pos, train_iter_neg, val_iter_pos, val_iter_neg, model, recons_pos, recons_neg,\
          mod_optimizer, scheduler, POS, NEG, num_epochs, logger=None):

    # Iterate through epochs
    bleu_best = -1
    itertrain_pos = iter(train_iter_pos)
    itertrain_neg = iter(train_iter_neg)
    num_batches = max(len(train_iter_pos), len(train_iter_neg))
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

            # POS RECONSTRUCTION LOSS
            pos_batch = next(itertrain_pos)
            pos_text = pos_batch.text.cuda() if use_gpu else pos_batch.text
            noisy_pos_text = noisy_sample(pos_text)

            scores, mean_pos, logvar_pos = model(noisy_pos_text, pos_text, 0, 0)

            # Remove <s> from trg and </s> from scores
            scores = scores[:-1]
            pos_text = pos_text[1:]

            # Reshape for loss function
            scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
            pos_text = pos_text.view(scores.size(0))

            # Pass through loss function
           
            KL_pos = -0.5 * torch.sum(1 + logvar_pos - (mean_pos ** 2) - torch.exp(logvar_pos))
            pos_loss = recons_pos(scores, pos_text) + KL_pos
            
            # NEG RECONSTRUCTION LOSS
            neg_batch = next(itertrain_neg)
            neg_text = neg_batch.text.cuda() if use_gpu else neg_batch.text
            noisy_neg_text = noisy_sample(neg_text)
            scores, mean_neg, logvar_neg = model(noisy_neg_text, neg_text, 1, 1)

            # Remove <s> from trg and </s> from scores
            scores = scores[:-1]
            neg_text = neg_text[1:]

            # Reshape for loss function
            scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
            neg_text = neg_text.view(scores.size(0))
            # Pass through loss function
            KL_neg = -0.5 * torch.sum(1 + logvar_neg - (mean_neg ** 2) - torch.exp(logvar_neg))
            neg_loss = recons_neg(scores, neg_text) + KL_neg

            total_recon_loss = pos_loss + neg_loss
            total_recon_loss.backward()
            recon_losses.update(total_recon_loss.data[0])

            # Clip gradient norms and step optimizer
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            mod_optimizer.step()
            mod_optimizer.zero_grad()

           

            


            # Log within epoch
            if i % 1000 == 10:
                logger.log('''Epoch [{e}/{num_e}]\t Batch [{b}/{num_b}]\t Recon_Loss: {r_l:.3f}'''.format(e=epoch+1, num_e=num_epochs, b=i, \
                                                                                                  num_b=num_batches, r_l=recon_losses.avg))

        # Log after each epoch
        logger.log('''Epoch [{e}/{num_e}] Complete. \t Recon_Loss: {r_l:.3f}'''.format(e=epoch+1, num_e=num_epochs, \
                                                                                                  r_l=recon_losses.avg))
        logger.log('Training time: {:.3f}'.format(time.time() - val_time))
