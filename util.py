from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import random
import nlpaug.augmenter.word as naw

from nlpaug.model.word_embs import fasttext as ft_model

from nlpaug.augmenter.word import WordEmbsAug
from nlpaug.model.word_embs import Fasttext
from gensim.models.fasttext import load_facebook_vectors

class WordEmbsAugBinary(WordEmbsAug):
    def __init__(self, model_path, action="substitute", aug_p=0.3):
        # Tải FastText binary gốc của Facebook bằng cách đúng
        model = Fasttext()
        model.model = load_facebook_vectors(model_path)

        # Khởi tạo augment dùng model đã load
        super().__init__(
            model_type='fasttext',
            model_path=None,  # Tránh nlpaug gọi lại
            model=model,
            action=action,
            aug_p=aug_p
        )

        
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class TextAugment:
    def __init__(self):
        self.aug = WordEmbsAugBinary(
            model_path='/kaggle/input/fasttext_de/pytorch/default/1/cc.de.300.bin',
            action="substitute",
            aug_p=0.3
        )

    def __call__(self, text):
        return [self.aug.augment(text)[0], self.aug.augment(text)[0]]


    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state