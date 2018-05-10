import itertools, os, sys, time
sys.path.append('../')

import torch
import torch.nn as nn
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()
from validate import Valid
from validate import validate
from utils import AverageMeter, exclusive_uniform, noisy_sample, distance_constrained_shuffle, Logger


def convert_batch(batch_text, field1, field2):
    string_list = []
    for a in range(batch_text.size()[1]):
        sentence = []
        for b in batch_text[:, a]:
            sentence.append(field1.vocab.itos[b.data[0]])
        string_list.append(sentence)
    for i, sentence in enumerate(string_list):
        for j, word in enumerate(sentence):
            batch_text[j][i].data[0] = field2.vocab.stoi[word]
    return batch_text


def train(train_iter_pos, train_iter_neg, val_iter_pos, val_iter_neg, model,discrim, recons_pos, recons_neg,d_loss,\
          mod_optimizer, discrim_optimizer, scheduler, POS, NEG, num_epochs, logger=None):

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
            for p in discrim.parameters():
                p.requires_grad = False
            mod_optimizer.zero_grad()

            # POS RECONSTRUCTION LOSS
            pos_batch = next(itertrain_pos)
            pos_text = pos_batch.text.cuda() if use_gpu else pos_batch.text
            noisy_pos_text = noisy_sample(pos_text)

            scores = model(noisy_pos_text, pos_text, 0, 0)

            # Remove <s> from trg and </s> from scores
            scores = scores[:-1]
            pos_text = pos_text[1:]

            # Reshape for loss function
            scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
            pos_text = pos_text.view(scores.size(0))

            # Pass through loss function
            pos_loss = recons_pos(scores, pos_text)


            # NEG RECONSTRUCTION LOSS
            neg_batch = next(itertrain_neg)
            neg_text = neg_batch.text.cuda() if use_gpu else neg_batch.text
            noisy_neg_text = noisy_sample(neg_text)
            scores = model(noisy_neg_text, neg_text, 1, 1)

            # Remove <s> from trg and </s> from scores
            scores = scores[:-1]
            neg_text = neg_text[1:]

            # Reshape for loss function
            scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
            neg_text = neg_text.view(scores.size(0))
            # Pass through loss function
            neg_loss = recons_neg(scores, neg_text)

            total_recon_loss = pos_loss + neg_loss
            total_recon_loss.backward()
            recon_losses.update(total_recon_loss.data[0])

            # Clip gradient norms and step optimizer
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            mod_optimizer.step()
            mod_optimizer.zero_grad()

            # CROSS-DOMAIN LOSS
            pos_text = pos_batch.text.cuda() if use_gpu else pos_batch.text
            pos_to_neg = model.greedy(pos_text, 0, 1)

            noisy_trans = noisy_sample(pos_to_neg)
            generated_pos = model.forward(noisy_trans, pos_text, 1, 0)

            # Remove <s> from trg and </s> from scores
            generated_pos = generated_pos[:-1]
            pos_text = pos_text[1:]

            # Reshape for loss function
            generated_pos = generated_pos.view(generated_pos.size(0) * generated_pos.size(1), generated_pos.size(2))
            pos_text = pos_text.view(generated_pos.size(0))

            # Pass through loss function
            cross_pos_loss = recons_pos(generated_pos, pos_text)

            neg_text = neg_batch.text.cuda() if use_gpu else neg_batch.text
            neg_to_pos = model.greedy(neg_text, 1, 0)

            noisy_trans = noisy_sample(neg_to_pos)
            generated_neg = model.forward(noisy_trans, neg_text,0, 1)

            # Remove <s> from trg and </s> from scores
            generated_neg = generated_neg[:-1]
            neg_text = neg_text[1:]

            # Reshape for loss function
            generated_neg = generated_neg.view(generated_neg.size(0) * generated_neg.size(1), generated_neg.size(2))
            neg_text = neg_text.view(generated_neg.size(0))


            # Pass through loss function
            cross_neg_loss = recons_neg(generated_neg, neg_text)


            total_cross_loss = cross_pos_loss + cross_neg_loss
            total_cross_loss.backward()
            cross_losses.update(total_cross_loss.data[0])

            # Clip gradient norms and step optimizer
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            mod_optimizer.step()

            # ADVERSARIAL TRAINING

            for p in discrim.parameters():
                p.requires_grad = True

            discrim_optimizer.zero_grad()
            pos_text = pos_batch.text.cuda() if use_gpu else pos_batch.text
            neg_text = neg_batch.text.cuda() if use_gpu else neg_batch.text
            z_hat_pos = discrim(model.encoder(pos_text, 0)[1][1].view(pos_text.size()[1], -1)).cuda()
            z_pos = Variable(torch.ones(pos_text.size()[1])).cuda()
            z_hat_neg = discrim(model.encoder(neg_text, 0)[1][1].view(neg_text.size()[1], -1)).cuda()
            z_neg = Variable(torch.ones(neg_text.size()[1])).cuda()
            real_loss = d_loss(z_hat_pos.squeeze(),z_pos)
            fake_loss = d_loss(z_hat_neg.squeeze(),z_neg)
            #step for discriminator
            disc_loss = real_loss + fake_loss

            disc_losses.update(disc_loss.data[0])
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm(discrim.parameters(), 1)
            discrim_optimizer.step()

            for p in discrim.parameters():
                p.requires_grad = False

            z_hat_neg_2 = discrim(model.encoder(neg_text, 0)[1][1].view(neg_text.size()[1], -1)).cuda()
            z_pos = Variable(torch.ones(neg_text.size()[1])).cuda()
            gen_loss = d_loss(z_hat_neg_2.squeeze(), z_pos)

            gen_losses.update(gen_loss.data[0])
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

            mod_optimizer.zero_grad()
            gen_loss.backward()
            mod_optimizer.step()


            # Log within epoch
            if i % 100 == 10:
                logger.log('''Epoch [{e}/{num_e}]\t Batch [{b}/{num_b}]\t Recon_Loss: {r_l:.3f}, Cross_Loss: {c_l:.3f}, Disc_Loss: {d_l:.3f}, Gen_Loss: {g_l:.3f}'''.format(e=epoch+1, num_e=num_epochs, b=i, \
                                                                                                  num_b=num_batches, r_l=recon_losses.avg, c_l = cross_losses.avg, d_l = disc_losses.avg, g_l = gen_losses.avg))

        # Log after each epoch
        logger.log('''Epoch [{e}/{num_e}] Complete. \t Recon_Loss: {r_l:.3f}, Cross_Loss: {c_l:.3f}, Disc_Loss: {d_l:.3f}, Gen_Loss: {g_l:.3f}'''.format(e=epoch+1, num_e=num_epochs, \
                                                                                                  r_l=recon_losses.avg, c_l = cross_losses.avg, d_l = disc_losses.avg, g_l = gen_losses.avg))
        logger.log('Training time: {:.3f}'.format(time.time() - val_time))

def convert_batch(batch_text, field1, field2):
    string_list = []
    for a in range(batch_text.size()[1]):
        sentence = []
        for b in batch_text[:, a]:
            sentence.append(field1.vocab.itos[b.data[0]])
        string_list.append(sentence)
    for i, sentence in enumerate(string_list):
        for j, word in enumerate(sentence):
            batch_text[j][i].data[0] = field2.vocab.stoi[word]
    return batch_text

def train_2(train_iter_pos, train_iter_neg, val_iter_pos, val_iter_neg, model, recons_pos, recons_neg,\
          mod_optimizer, scheduler, POS, NEG, TEXT, num_epochs, logger=None):

    # Iterate through epochs
    bleu_best = -1
    itertrain_pos = iter(train_iter_pos)
    itertrain_neg = iter(train_iter_neg)
    num_batches = max(len(train_iter_pos), len(train_iter_neg))
    for epoch in range(num_epochs):

        # Validate model with BLEU
        start_time = time.time() # timer
        bleu_val = validate_2(val_iter_pos, val_iter_neg, model, POS, NEG, TEXT)
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
            pos_text = convert_batch(pos_batch.text, POS, TEXT)
            pos_text = pos_text.cuda() if use_gpu else pos_text
            #noisy_pos_text = noisy_sample(pos_text)
            print(pos_text)

            scores = model(pos_text, pos_text,  0)

            # Remove <s> from trg and </s> from scores
            scores = scores[:-1]
            pos_text = pos_text[1:]

            # Reshape for loss function
            scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
            pos_text = pos_text.view(scores.size(0))

            # Pass through loss function
            pos_loss = recons_pos(scores, pos_text)


            # NEG RECONSTRUCTION LOSS
            neg_batch = next(itertrain_neg)
            neg_text = convert_batch(neg_batch.text, NEG, TEXT)
            neg_text = neg_text.cuda() if use_gpu else neg_text
            #noisy_neg_text = noisy_sample(neg_text)
            scores = model(neg_text, neg_text, 1)

            # Remove <s> from trg and </s> from scores
            scores = scores[:-1]
            neg_text = neg_text[1:]

            # Reshape for loss function
            scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
            neg_text = neg_text.view(scores.size(0))
            # Pass through loss function
            neg_loss = recons_neg(scores, neg_text)

            total_recon_loss = pos_loss + neg_loss
            total_recon_loss.backward()
            recon_losses.update(total_recon_loss.data[0])

            # Clip gradient norms and step optimizer
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            mod_optimizer.step()
            mod_optimizer.zero_grad()


            # Log within epoch
            if i % 100 == 10:
                logger.log('''Epoch [{e}/{num_e}]\t Batch [{b}/{num_b}]\t Recon_Loss: {r_l:.3f}'''.format(e=epoch+1, num_e=num_epochs, b=i, \
                                                                                                  num_b=num_batches, r_l=recon_losses.avg))

        # Log after each epoch
        logger.log('''Epoch [{e}/{num_e}] Complete. \t Recon_Loss: {r_l:.3f}'''.format(e=epoch+1, num_e=num_epochs, \
                                                                                                  r_l=recon_losses.avg))
        logger.log('Training time: {:.3f}'.format(time.time() - val_time))
