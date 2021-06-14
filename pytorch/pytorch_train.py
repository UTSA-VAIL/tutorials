import sys
import os
import pickle
import torch
import torch.nn as nn
import multiprocessing as mp
import numpy as np
import cv2
import torchvision.models as models
from torch.utils.data import DataLoader 
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def train_imgnet_cpu(model, optim, cce, data):
    device = torch.device('cpu')
    model.to(device)
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    for epoch in range(10):

        for idx, batch in enumerate(data):
            inputs, labels = batch
            inputs.to(device)
            optim.zero_grad()
            outputs = model(inputs)
            loss = cce(outputs,labels)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.detach().item(), inputs.size(0))        
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            print(losses)
            print(top5)
            optim.zero_grad()
            loss.backward()
            optim.step()

def train_main(dataset : str, is_gpu, data_path):
    if dataset == 'imagenet':
        if not is_gpu:
            model = models.vgg19(pretrained=False)
            optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
            cce = nn.CrossEntropyLoss()
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

            train_dataset = datasets.ImageFolder(
                data_path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=128, shuffle=True,
                num_workers=4, pin_memory=True)
            train_imgnet_cpu(model, optimizer, cce, train_loader)



if __name__ == '__main__':
    train_main('imagenet', False, '~/Downloads/ILSVRC/Data/DET/train/ILSVRC2013_train')
