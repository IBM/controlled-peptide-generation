import os
import codecs
import torch
import numpy as np
from functools import reduce
import operator


def describe(t):  # t could be numpy or torch tensor.
    t = t.data if isinstance(t, torch.autograd.Variable) else t
    s = '{:17s} {:8s} [{:.4f} , {:.4f}] m+-s = {:.4f} +- {:.4f}'
    ttype = 'np.{}'.format(t.dtype) if type(t) == np.ndarray else str(t.type()).replace('ensor', '')
    si = 'x'.join(map(str, t.shape if isinstance(t, np.ndarray) else t.size()))
    return s.format(ttype, si, t.min(), t.max(), t.mean(), t.std())


def write_gen_samples(samples, fn, c_lab=None):
    """ samples: list of strings. c_lab (optional): tensor of same size. """
    fn_dir = os.path.dirname(fn)
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)

    size = len(samples)
    with open(fn, 'w+') as f:
        if c_lab is not None:
            print("Saving %d samples with labels" % size)
            assert c_lab.nelement() == size, 'sizes dont match'
            f.writelines(['label: {}\n{}\n'.format(y, s) for y, s in zip(c_lab, samples)])
        else:
            print("Saving %d samples without labels" % size)
            f.write('\n'.join(samples) + '\n')


def write_interpsamples(samples, fn, c_lab=False):
    raise Exception('Reimplement this function like write_gen_samples(), use minibatch')


def write_samezsamples(samples, samples2, fn, fn2, lab=False):
    raise Exception('Reimplement this function like write_gen_samples(), use minibatch')


def save_vocab(vocab, fn):
    check_dir_exists(fn)
    with codecs.open(fn, "w", "utf-8") as f:
        for word, ix in vocab.stoi.items():
            f.write(word + " " + str(ix) + "\n")
    print('Saved vocab to ' + fn)


# Linearly interpolate between start and end val depending on current iteration
def interpolate(start_val, end_val, start_iter, end_iter, current_iter):
    if current_iter < start_iter:
        return start_val
    elif current_iter >= end_iter:
        return end_val
    else:
        return start_val + (end_val - start_val) * (current_iter - start_iter) / (end_iter - start_iter)


def anneal(cfgan, it):
    return interpolate(cfgan.start.val, cfgan.end.val, cfgan.start.iter, cfgan.end.iter, it)


def check_dir_exists(fn):
    fn_dir = os.path.dirname(fn)
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def scale_and_clamp(dist, w, clamp_val=None):
    rescaled = dist * w  # w = 1/scale
    if clamp_val and rescaled > clamp_val:
        return clamp_val
    else:
        return rescaled
