import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from itertools import chain
import random
import torch.nn.functional as F
from torch.autograd import Variable
from model import NetworkCELEBALFW, NetworkRESNET, NetworkCIFAR
from datasets import LFWClassificationDataset, CelebADataset, LFWPairsDataset
from contrastive_center_loss import ContrastiveCenterLoss
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
from resnet import resnet_face

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--thresh', type=float, default=3.0, help='Threshold')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTSCIFAR', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--dataset', type=str, default='cifar', help='dataset name')
parser.add_argument('--n_features', type=int, default=128, help='num of face features')

parser.add_argument('--alpha', type=float, default=0.5,
                    help='learning rate of class center (default: 0.5)')
parser.add_argument('--ccl_loss', action='store_true', default=False, help='use contrastive Loss')
parser.add_argument('--lambda-c', type=float, default=1e05,
                    help='weight parameter of center loss (default: 1.0) It was 0.5')
parser.add_argument('--aucflag', action='store_true', default=False, help='auc score flag')
parser.add_argument('--accflag', action='store_true', default=False, help='use acc flag')
parser.add_argument('--celeb_cut', action='store_true', default=False, help='cut to number of classes')
parser.add_argument('--celeb_class_size', type=int, default=5000, help='Celeb Class size')
parser.add_argument('--resnet_layers', type=int, default=50, help='number of resnet layers')
parser.add_argument('--image_size', type=int, default=64, help='image_size')

parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    if args.dataset == 'cifar':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

        numclasses = CIFAR_CLASSES

    elif args.dataset == 'lfwdeepfunnel':
        train_transform, valid_transform = utils._data_transforms_lfw(args)
        lfw_classification_dataset_train = LFWClassificationDataset(train=True, pil=True, size=args.image_size,
                                                                    image_transform=train_transform)
        lfw_classification_dataset_test = LFWClassificationDataset(train=False, pil=True, size=args.image_size,
                                                                   image_transform=train_transform)

        le = preprocessing.LabelEncoder()

        train_labels = list(lfw_classification_dataset_train.targets)
        test_labels = list(lfw_classification_dataset_test.targets)

        all_labels = list(chain(train_labels, test_labels))
        encoded_labels = list(le.fit_transform(all_labels))

        all_imgs = list(chain(lfw_classification_dataset_train.images, lfw_classification_dataset_test.images))

        df = pd.DataFrame(list(zip(all_imgs, encoded_labels)), columns=['images', 'enc_lbls'])

        counter = Counter(encoded_labels)
        d = dict(counter)
        df['occurence'] = df['enc_lbls'].apply(lambda x: d[x])
        cut_df = df[df['occurence'] > args.thresh].reset_index(drop=True)

        if args.aucflag:
            le_enc = preprocessing.LabelEncoder()

            lfw_classification_dataset_train.images = list(cut_df['images'].values)
            lfw_classification_dataset_train.targets = list(le_enc.fit_transform(list(cut_df['enc_lbls'].values)))
            numclasses = len(le_enc.classes_)

            valid_data = LFWPairsDataset(train=False, pil=True, size=args.image_size, image_transform=valid_transform,
                                         root=args.data)

            train_queue = torch.utils.data.DataLoader(
                lfw_classification_dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

            valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
        elif args.accflag:
            X_train_imgs, X_test_imgs, y_train, y_test = train_test_split(cut_df['images'], cut_df['enc_lbls'],
                                                                          test_size=0.25, stratify=cut_df['enc_lbls'], random_state=42)
            le_enc = preprocessing.LabelEncoder()

            lfw_classification_dataset_train.images, lfw_classification_dataset_test.images = list(X_train_imgs.values),\
                                                                                              list(X_test_imgs.values)
            lfw_classification_dataset_train.targets = list(le_enc.fit_transform(list(y_train.values)))
            lfw_classification_dataset_test.targets = list(le_enc.transform(list(y_test.values)))

            numclasses = len(le_enc.classes_)

            train_queue = torch.utils.data.DataLoader(
                lfw_classification_dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

            valid_queue = torch.utils.data.DataLoader(
                lfw_classification_dataset_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    elif args.dataset == 'celebA':
        train_transform, valid_transform = utils._data_transforms_lfw(args)
        celeb_data_train = CelebADataset(flag=0, pil=True, size=args.image_size, image_transform=train_transform,
                                         num_img=15)
        celeb_data_test = CelebADataset(flag=1, pil=True, size=args.image_size, image_transform=train_transform,
                                        num_img=15)

        all_labels = list(chain(celeb_data_train.targets, celeb_data_test.targets))
        all_imgs = list(chain(celeb_data_train.imgs, celeb_data_test.imgs))
        df = pd.DataFrame(list(zip(all_imgs, all_labels)), columns=['images', 'labels'])

        counter = Counter(all_labels)
        d = dict(counter)
        frqs = counter.values()
        df['occurence'] = df['labels'].apply(lambda x: d[x])
        threshold = sum(frqs) / len(frqs)

        cut_df = df[df['occurence'] > threshold].reset_index(drop=True)

        if args.celeb_cut:
            d_cut = dict(Counter(list(cut_df['labels'].values)))
            sample_classes = random.sample(list(d_cut.keys()), args.celeb_class_size)
            cut_df = cut_df.loc[cut_df['labels'].isin(sample_classes)].reset_index(drop=True)

        X_train_imgs, X_test_imgs, y_train, y_test = train_test_split(cut_df['images'], cut_df['labels'],
                                                                      test_size=0.25, stratify=cut_df['labels'],
                                                                      random_state=42)
        le_enc = preprocessing.LabelEncoder()

        celeb_data_train.imgs = list(X_train_imgs.values)
        celeb_data_train.targets = list(le_enc.fit_transform(list(y_train.values)))
        numclasses = len(le_enc.classes_)

        train_queue = torch.utils.data.DataLoader(
            celeb_data_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

        if args.aucflag:
            valid_data = LFWPairsDataset(train=False, pil=True, size=args.image_size, image_transform=valid_transform,
                                         root=args.data)

            valid_queue_auc = torch.utils.data.DataLoader(
                valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

        if args.accflag:
            celeb_data_test.imgs = list(X_test_imgs.values)
            celeb_data_test.targets = list(le_enc.transform(list(y_test.values)))
            numclasses = len(le_enc.classes_)

            valid_queue_acc = torch.utils.data.DataLoader(
                celeb_data_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    if 'DARTS' in args.arch:
        genotype = eval("genotypes.%s" % args.arch)

        if args.dataset == 'cifar':
            model = NetworkCIFAR(args.init_channels, numclasses, args.layers, args.auxiliary, genotype)
        else:
            model = NetworkCELEBALFW(args.init_channels, numclasses, args.layers, genotype, args.drop_path_prob, args.ccl_loss)

        if args.ccl_loss:
            center_loss = ContrastiveCenterLoss(dim_hidden=model.classifier.in_features, num_classes=numclasses,
                                            lambda_c=args.lambda_c)
    elif args.arch == 'RESNET':
        #net = NetworkRESNET(args.resnet_layers, True)
        model = resnet_face(ccl_flag=args.ccl_loss, use_se=False, n_features=args.n_features, numclasses=numclasses)
        if args.ccl_loss:
            center_loss = ContrastiveCenterLoss(dim_hidden=args.n_features, num_classes=numclasses,
                                            lambda_c=args.lambda_c)

    if args.ccl_loss:
        nll_loss = nn.NLLLoss()
        nll_loss, center_loss = nll_loss.cuda(), center_loss.cuda()
        criterion = [nll_loss, center_loss]
    else:
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    global model_params, master_params

    if args.fp16:
        model = network_to_half(model)

    # if args.fp16:
    #     model_params, master_params = prep_param_lists(model)
    # else:
    #     master_params = model.parameters()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.ccl_loss:
        optimizer_c = torch.optim.SGD(center_loss.parameters(), lr=args.alpha)
        optimizer_final = [optimizer, optimizer_c]
    else:
        optimizer_final = optimizer

    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale)
        optimizer_apex = [optimizer, FP16_Optimizer(optimizer_c,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale)]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    train_accuracies = []
    val_accuracies = []

    train_accuracies_auc = []
    val_accuracies_auc = []

    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        if args.fp16:
            train_acc, train_obj = train_apex(train_queue, model, criterion, optimizer_apex)
        else:
            train_acc, train_obj = train(train_queue, model, criterion, optimizer_final)

        logging.info('train_acc %f', train_acc)
        train_accuracies.append(train_acc)

        if args.accflag:
            valid_acc, valid_obj = infer(valid_queue_acc, model, criterion)
            logging.info('valid_acc %f', valid_acc)
            val_accuracies.append(valid_acc)

        if args.aucflag:
            valid_auc, valid_tpr_star = infer_auc(valid_queue_auc, model)
            logging.info('valid: auc={auc}, tpr_star={tpr_star}'.format(auc=valid_auc, tpr_star=valid_tpr_star))
            val_accuracies_auc.append(valid_auc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        print(args.save)

    print("Train Accuracies are " + str(train_accuracies)[1:-1])
    print("Validation Accuracies are " + str(val_accuracies)[1:-1])
    print("Train AUC Scores are " + str(train_accuracies_auc)[1:-1])
    print("Validation AUC Scores are " + str(val_accuracies_auc)[1:-1])


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.43920362 * 255, 0.38309247 * 255, 0.34243814 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.29700514 * 255, 0.27357439 * 255, 0.26827176 * 255]).cuda().view(1, 3, 1, 1)
        if args.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(async=True)
            self.next_target = self.next_target.cuda(async=True)
            if args.fp16:
                self.next_input = self.next_input.half()
            else:
                self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def train_apex(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    prefetcher = data_prefetcher(train_queue)

    input, target = prefetcher.next()
    i = -1

    while input is not None:
        i += 1
        input = input.cuda()
        target = target.cuda(async=True)

        if args.ccl_loss:
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            feat, logits = model(input)
            loss1 = criterion[0](logits, target)
            loss2 = criterion[1](target, feat)
            loss = loss1 + loss2
            loss = loss * args.static_loss_scale
            if args.fp16:
                #optimizer[0].backward(loss1, update_master_grads=False)
                #optimizer[0].backward(loss2, update_master_grads=False)
                #optimizer[0].update_master_grads()
                optimizer[0].backward(loss1)
                optimizer[1].backward(loss2)
            else:
                loss.backward()
            optimizer[0].step()
            optimizer[1].step()
            #print(A.sjape)
        else:
            optimizer.zero_grad()
            logits, _ = model(input)
            loss = criterion(logits, target)
            loss = loss * args.static_loss_scale
            loss.backward()
            optimizer.step()

        if float(torch.__version__[0:3]) > 0.35:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)

        if float(torch.__version__[0:3]) > 0.35:
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
        else:
            objs.update(loss.data[0], n)
            top1.update(prec1.data[0], n)
            top5.update(prec5.data[0], n)

        # if step % args.report_freq == 0:
        #     logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda(async=True)

        if args.ccl_loss:
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            feat, logits = model(input)
            loss = criterion[0](logits, target) + criterion[1](target, feat)
            loss = loss * args.static_loss_scale
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()
        else:
            optimizer.zero_grad()
            logits, _ = model(input)
            loss = criterion(logits, target)
            loss = loss * args.static_loss_scale
            loss.backward()
            optimizer.step()

        if float(torch.__version__[0:3]) > 0.35:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)

        if float(torch.__version__[0:3]) > 0.35:
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
        else:
            objs.update(loss.data[0], n)
            top1.update(prec1.data[0], n)
            top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
      for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(async=True)

        feat, logits = model(input)
        loss = criterion[0](logits, target) + criterion[1](target, feat)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)

        if float(torch.__version__[0:3]) > 0.35:
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
        else:
            objs.update(loss.data[0], n)
            top1.update(prec1.data[0], n)
            top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

      return top1.avg, objs.avg


def infer_auc(valid_queue, model):
    positive_scores = []
    negative_scores = []
    model.eval()

    with torch.no_grad():
        for step, (pair, target) in enumerate(valid_queue):
            image_a, image_b = pair
            target = target.numpy()

            image_a = image_a.cuda()
            image_b = image_b.cuda()

            image_a_features = model(image_a)[0].squeeze(-1).squeeze(-1)
            image_b_features = model(image_b)[0].squeeze(-1).squeeze(-1)

            scores = F.cosine_similarity(image_a_features, image_b_features).data.cpu().numpy()

            positive_scores.append(scores[target == 1])
            negative_scores.append(scores[target == 0])

        auc = utils.auc(positive_scores, negative_scores)
        tpr_star = utils.tpr_star(positive_scores, negative_scores)
        logging.info("validation metrics: auc @ {auc}, tpr_star @ {tpr_star}".format(auc=auc, tpr_star=tpr_star))

        return auc, tpr_star


if __name__ == '__main__':
    main()