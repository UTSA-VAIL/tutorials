import sys
import os
import pickle
from tensorflow.keras import optimizers
import torch
import torch.nn as nn
from torchsummary import summary
import multiprocessing as mp
import numpy as np
import cv2
import torchvision.models as models
from torch.utils.data import DataLoader 

class Loader:

    def __init__(self):
        pass

    def __call__(self, img : str, gt : str):
        pass


class TorchData(torch.utils.data.Dataset):

    def __init__(self, mapped_paths : dict):
        pass


    def __len__(self) -> int:
        """ an override of length """
        pass

    def __getitem__(self, index : int):
        """an override of the getitem from torch Dataset"""
        pass

def get_preds(outputs):
    outputs = 
    pr = outputs.detach().numpy()


def training_pipeline(model, optim, optim, cce, data):
    device = torch.device('cpu')
    model.to(device)
    model.train()
    for epoch in range(10):

        for idx, batch in enumerate(data):
            inputs, labels = batch
            inputs.to(device)
            optim.zero_grad()
            outputs = model(inputs)
            preds = get_preds(outputs)


def train_main():
    model = models.vgg19(pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.99, learning_rate=-1e3)
    cce = nn.CrossEntropyLoss()
    train_data = TorchData(dict())
    train = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    training_pipeline(model, optimizer, cce, train)



if __name__ == '__main__':
    train_main()