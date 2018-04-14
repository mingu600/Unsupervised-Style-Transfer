import itertools, os, re
import tempfile, subprocess
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext
import numpy as np
from torchtext.vocab import Vectors, GloVe
use_gpu = torch.cuda.is_available()
import random


# From https://stackoverflow.com/questions/30747690/controlling-distance-of-shuffling
def exclusive_uniform(a, b):
    "returns a random value in the interval  [a, b)"
    return a+(b-a)*random.random()

def distance_constrained_shuffle(sequence, distance,
                                 randmoveforward = exclusive_uniform):
    def sort_criterion(enumerate_tuple):
        """
        returns the index plus a random offset,
        such that the result can overtake at most 'distance' elements
        """
        indx, value = enumerate_tuple
        return indx + randmoveforward(0, distance+1)

    # get enumerated, shuffled list
    enumerated_result = sorted(enumerate(sequence), key = sort_criterion)
    # remove enumeration
    result = [x for i, x in enumerated_result]
    return result

def noisy_sample(input_seq, p_wd=0.1, k=3):
    length = input_seq.size()[0]
    if length < 3:
        return input_seq
    words_kept = np.where((np.random.binomial(1, p_wd,  length- 2) ==0) == 1)[0] + 1
    return input_seq[np.array([0] + distance_constrained_shuffle(words_kept, k) + [length - 1]), :]

class Logger():
    '''Prints to a log file and to standard output'''
    def __init__(self, path):
        if os.path.exists(path):
            self.path = path
        else:
            raise Exception('path does not exist')

    def log(self, info, stdout=True):
        with open(os.path.join(self.path, 'log.log'), 'a') as f:
            print(info, file=f)
        if stdout:
            print(info)

    def save_model(self, model_dict):
        #with open(os.path.join(self.path, 'model.pkl'), 'w') as f:
        torch.save(model_dict, os.path.join(self.path, 'model.pkl'))

class AverageMeter():
    '''Computes and stores the average and current value.
       Taken from the PyTorch ImageNet tutorial'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count
