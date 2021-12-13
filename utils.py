import os
import numpy as np
import torch
import shutil
import bisect
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, roc_curve


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def auc(positive_scores, negative_scores):
    positive_scores = np.concatenate(positive_scores)
    negative_scores = np.concatenate(negative_scores)
    positive_labels = np.ones(positive_scores.shape)
    negative_labels = np.zeros(negative_scores.shape)
    scores = np.concatenate((positive_scores, negative_scores))
    labels = np.concatenate((positive_labels, negative_labels))
    return roc_auc_score(labels, scores)


def tpr_star(positive_scores, negative_scores, star_fpr=0.01):
    positive_scores = np.concatenate(positive_scores)
    negative_scores = np.concatenate(negative_scores)
    positive_labels = np.ones(positive_scores.shape)
    negative_labels = np.zeros(negative_scores.shape)
    scores = np.concatenate((positive_scores, negative_scores))
    labels = np.concatenate((positive_labels, negative_labels))
    fpr, tpr, _ = roc_curve(labels, scores)
    star_idx = bisect.bisect_left(fpr, star_fpr)
    fpr_left = 0 if star_idx <= 0 else fpr[star_idx - 1]
    tpr_left = 0 if star_idx <= 0 else tpr[star_idx - 1]
    fpr_right = 1 if star_idx == len(fpr) else fpr[star_idx]
    tpr_right = 1 if star_idx == len(tpr) else tpr[star_idx]
    dtpr_dfpr = (tpr_right - tpr_left) / (fpr_right - fpr_left)
    return tpr_left + (star_fpr - fpr_left) * dtpr_dfpr


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6


def _data_transforms_lfw(args):
    LFW_MEAN = [0.43920362, 0.38309247, 0.34243814]
    LFW_STD = [0.29700514, 0.27357439, 0.26827176]

    train_transform = transforms.Compose([
        transforms.RandomCrop(args.image_size, padding=4),
        # TODO: IMPROVEMENT: Find the sweet spot for image/batch size~
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(LFW_MEAN, LFW_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(LFW_MEAN, LFW_STD),
    ])
    return train_transform, valid_transform


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.HalfTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        # if args.fp16:
        #     mask = Variable(torch.cuda.HalfTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        # else:
        #     mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
