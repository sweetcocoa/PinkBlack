import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
from time import time
from datetime import datetime
from collections import defaultdict

from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json

from .PinkModule.logging import *


class AverageMeter(object):
    """Computes and stores the average and current value

        Original Code :: https://github.com/pytorch/example/mnist
    """

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


def cal_accuracy(pred, target):
    pred = torch.max(pred, 1)[1]
    corrects = torch.sum(pred == target).float()
    return corrects / pred.size(0)


class Trainer:
    def __init__(self, net,
                 criterion=None,
                 metric=cal_accuracy,
                 dataloader=None,
                 train_dataloader=None,
                 val_dataloader=None,
                 optimizer=None,
                 lr_scheduler=None,
                 logdir="./pinkblack_autolog/",
                 ckpt="./ckpt/ckpt.pth",
                 clip_gradient_norm=False,
                 ):
        """
        :param net: nn.Module Network. __call__(*batch_x)
        :param criterion: loss function. __call__(prediction, *batch_y)
        :param metric: metric function __call__(prediction, *batch_y).
        *note* : bigger is better.
        :param dataloader: 'train':train_dataloader, 'val':val_dataloader
        :param train_dataloader:
        :param val_dataloader:
        :param optimizer: optimizer (torch.optim)
        :param lr_scheduler:
        :param logdir: tensorboardX log
        :param ckpt:
        """

        self.net = net
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.metric = metric

        if dataloader is not None and isinstance(dataloader, dict):
            self.dataloader = dataloader
        elif train_dataloader is not None and val_dataloader is not None:
            self.dataloader = {'train': train_dataloader,
                               'val': val_dataloader}
        else:
            raise RuntimeError("Init Trainer :: Two dataloaders are needed!")

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters())) if optimizer is None else optimizer
        self.lr_scheduler = lr_scheduler

        self.ckpt = ckpt

        self.config = defaultdict(float)
        self.config['max_val_metric'] = -1e8
        self.config['logdir'] = logdir
        self.config['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config['clip_gradient_norm'] = clip_gradient_norm

        self.device = torch.device("cpu")
        for param in self.net.parameters():
            self.device = param.device
            break

        if self.config['logdir'] is not None:
            try:
                from tensorboardX import SummaryWriter
                self.logger = SummaryWriter(self.config['logdir'])
            except ImportError:
                self.logger = None
        else:
            self.logger = None

    def save(self, f=None):
        if f is None:
            f = self.ckpt
        os.makedirs(os.path.dirname(f), exist_ok=True)
        if hasattr(self.net, 'module'):
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()
        torch.save(state_dict, f)
        torch.save(self.optimizer.state_dict(), f + ".optimizer")
        with open(f + ".config", "w") as f:
            json.dump(self.config, f)

    def load(self, f=None):
        if f is None:
            f = self.ckpt

        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(torch.load(f, map_location=self.device))
        else:
            self.net.load_state_dict(torch.load(f, map_location=self.device))

        if os.path.exists(f + ".config"):
            with open(f + ".config", "r") as fp:
                dic = json.loads(fp.read())
            self.config = defaultdict(float, dic)
            print("Loaded,", self.config)

        if os.path.exists(f + ".optimizer"):
            self.optimizer.load_state_dict(torch.load(f + ".optimizer"))

        if self.config['logdir'] is not None:
            try:
                from tensorboardX import SummaryWriter
                self.logger = SummaryWriter(self.config['logdir'])
            except ImportError:
                self.logger = None
        else:
            self.logger = None

    def train(self, epoch, phases=['train', 'val']):
        kwarg_list = ['epoch', 'train_loss', 'train_metric',
                      'val loss', 'val metric', 'time']

        print_row(kwarg_list=['']*len(kwarg_list), pad='-')
        print_row(kwarg_list=kwarg_list, pad=' ')
        print_row(kwarg_list=['']*len(kwarg_list), pad='-')

        start_epoch = int(self.config['epoch'])
        for ep in range(start_epoch + 1, start_epoch + epoch + 1):
            start_time = time()

            for phase in phases:
                self.config[f'{phase}_loss'], self.config[f'{phase}_metric'] = self._train(phase)

            self.config['epoch'] += 1
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(self.config['val_loss'])
                else:
                    self.lr_scheduler.step()

            ep_str = str(ep)
            if self.config['max_val_metric'] < self.config['val_metric']:
                self.config['max_val_metric'] = self.config['val_metric']
                self.save()
                ep_str = (str(ep)) + '-best'

            elapsed_time = time() - start_time
            if self.logger is not None:
                self.logger.add_scalars(f"{self.config['timestamp']}/loss", {'train' : self.config['train_loss'],
                                                        'val': self.config['val_loss']}, ep)
                self.logger.add_scalars(f"{self.config['timestamp']}/metric", {'train' : self.config['train_metric'],
                                                        'val': self.config['val_metric']}, ep)
                self.logger.add_scalar(f"{self.config['timestamp']}/time", elapsed_time, ep)

            print_row(kwarg_list=[ep_str, self.config['train_loss'], self.config['train_metric'],
                                  self.config['val_loss'], self.config['val_metric'], elapsed_time], pad=' ')
            print_row(kwarg_list=['']*len(kwarg_list), pad='-')

    def _train(self, phase):

        running_loss = AverageMeter()
        running_metric = AverageMeter()

        if phase == 'train':
            self.net.train()
            dataloader = self.dataloader['train']
        elif phase == "val":
            self.net.eval()
            dataloader = self.dataloader['val']

        for batch_x, batch_y in tqdm(dataloader, leave=False):
            self.optimizer.zero_grad()

            if isinstance(batch_x, list):
                batch_x = [x.to(self.device) for x in batch_x]
            else:
                batch_x = [batch_x.to(self.device)]

            if isinstance(batch_y, list):
                batch_y = [y.to(self.device) for y in batch_y]
            else:
                batch_y = [batch_y.to(self.device)]

            with torch.set_grad_enabled(phase == "train"):
                outputs = self.net(*batch_x)
                loss = self.criterion(outputs, *batch_y)

                if phase == "train":
                    loss.backward()
                    self.optimizer.step()

                    if self.config['clip_gradient_norm']:
                        clip_grad_norm_(self.net.parameters(), self.config['clip_gradient_norm'])

            with torch.no_grad():
                metric = self.metric(outputs, *batch_y)

            running_loss.update(loss.item(), batch_x[0].size(0))
            running_metric.update(metric.item(), batch_x[0].size(0))

        return running_loss.avg, running_metric.avg

if __name__ == "__main__":
    # demo.py

    import torch
    import torchvision.utils as vutils
    import numpy as np
    import torchvision.models as models
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from tensorboardX import SummaryWriter
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))  
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return x

    resnet18 = Net().cuda()

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST('mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('mnist', train=False, transform=transform)
    train_dataloader = DataLoader(train_dataset, pin_memory=True, num_workers=4, shuffle=True, batch_size=32)
    test_dataloader = DataLoader(test_dataset, pin_memory=True, num_workers=4, batch_size=32, shuffle=True)

    trainer = Trainer(resnet18, train_dataloader=train_dataloader, val_dataloader=test_dataloader)
    trainer.new_train(10)
