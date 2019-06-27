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
from .PinkModule.swa_batchnorm_utils import *
import logging
import pandas as pd

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
                 train_dataloader=None,
                 val_dataloader=None,
                 test_dataloader=None,
                 optimizer=None,
                 lr_scheduler=None,
                 logdir="./pinkblack_autolog/",
                 ckpt="./ckpt/ckpt.pth",
                 clip_gradient_norm=False,
                 is_data_dict=False,
                 ):
        """
        :param net: nn.Module Network. __call__(*batch_x)
        :param criterion: loss function. __call__(prediction, *batch_y)
        :param metric: metric function __call__(prediction, *batch_y).
        *note* : bigger is better.
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

        self.dataloader = dict()
        if train_dataloader is not None:
            self.dataloader['train'] = train_dataloader
        if val_dataloader is not None:
            self.dataloader['val'] = val_dataloader
        if test_dataloader is not None:
            self.dataloader['test'] = test_dataloader

        if train_dataloader is None or val_dataloader is None:
            logging.warning("Init Trainer :: Two dataloaders are needed!")

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters())) if optimizer is None else optimizer
        self.lr_scheduler = lr_scheduler

        self.ckpt = ckpt

        self.config = defaultdict(float)
        self.config['max_train_metric'] = -1e8
        self.config['max_val_metric'] = -1e8
        self.config['max_test_metric'] = -1e8
        self.config['logdir'] = logdir
        self.config['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config['clip_gradient_norm'] = clip_gradient_norm
        self.config['is_data_dict'] = is_data_dict

        self.dataframe = pd.DataFrame()

        self.device = torch.device("cpu")
        for param in self.net.parameters():
            self.device = param.device
            break

        if self.device == torch.device("cpu"):
            logging.warning("Init Trainer :: Do you really want to train the network on CPU instead of CUDA?")

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
        if isinstance(self.net, nn.DataParallel):
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()
        torch.save(state_dict, f)
        torch.save(self.optimizer.state_dict(), f + ".optimizer")
        with open(f + ".config", "w") as fp:
            json.dump(self.config, fp)

        self.dataframe.to_csv(f + ".csv", float_format="%.4f", index=False)

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

        if os.path.exists(f + ".csv"):
            self.dataframe = pd.read_csv(f + ".csv")

        if self.config['logdir'] is not None:
            try:
                from tensorboardX import SummaryWriter
                self.logger = SummaryWriter(self.config['logdir'])
            except ImportError:
                self.logger = None
        else:
            self.logger = None

    def train(self, epoch=None,
              phases=None,
              step=None,
              validation_interval=1,
              save_every_validation=False):

        """
        :param epoch: train dataloader를 순회할 횟수
        :param phases: ['train', 'val', 'test'] 중 필요하지 않은 phase를 뺄 수 있다.
        >> trainer.train(1, phases=['val'])
        ... validation without training ...

        :param step: epoch이 아닌 step을 훈련단위로 할 때의 총 step 수.
        :param validation_interval: step이 훈련단위일때 validation 간격
        :param save_every_validation: True이면, validation마다 checkpoint를 저장한다.
        :return: None
        """
        if phases is None:
            phases = list(self.dataloader.keys())

        if epoch is None and step is None:
            raise ValueError("PinkBlack.trainer :: epoch or step should be specified.")

        train_unit = 'epoch' if step is None else 'step'
        num_unit = epoch if step is None else step
        validation_interval = 1 if validation_interval <= 0 else validation_interval

        kwarg_list = [train_unit]
        for phase in phases:
            kwarg_list += [f"{phase}_loss", f"{phase}_metric"]
        kwarg_list += ['lr', 'time']

        print_row(kwarg_list=['']*len(kwarg_list), pad='-')
        print_row(kwarg_list=kwarg_list, pad=' ')
        print_row(kwarg_list=['']*len(kwarg_list), pad='-')

        start = int(self.config[train_unit])

        for i in range(start + 1, start + num_unit + 1, validation_interval):
            start_time = time()
            if train_unit == "epoch":
                for phase in phases:
                    self.config[f'{phase}_loss'], self.config[f'{phase}_metric'] = self._train(phase, num_steps=len(self.dataloader[phase]))
                self.config[train_unit] += 1
            else:
                for phase in phases:
                    if phase == "train":
                        num_steps = min((start + num_unit + 1 - i), validation_interval)
                        self.config[train_unit] += num_steps
                    else:  # phase == "val":
                        num_steps = len(self.dataloader[phase])
                    self.config[f'{phase}_loss'], self.config[f'{phase}_metric'] = self._train(phase, num_steps=num_steps)

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(self.config['val_metric'])
                else:
                    self.lr_scheduler.step()

            i_str = str(self.config[train_unit])
            is_best = self.config['max_val_metric'] < self.config['val_metric']
            if is_best:
                for phase in phases:
                    self.config[f'max_{phase}_metric'] = max(self.config[f'max_{phase}_metric'],
                                                             self.config[f"{phase}_metric"])
                i_str = (str(self.config[train_unit])) + '-best'

            elapsed_time = time() - start_time
            if self.logger is not None:
                _loss, _metric = {}, {}
                for phase in phases:
                    _loss[phase] = self.config[f"{phase}_loss"]
                    _metric[phase] = self.config[f"{phase}_metric"]

                self.logger.add_scalars(f"{self.config['timestamp']}/loss", _loss, i)
                self.logger.add_scalars(f"{self.config['timestamp']}/metric", _metric, i)
                self.logger.add_scalars(f"{self.config['timestamp']}/time", elapsed_time, i)
                self.logger.add_scalars(f"{self.config['timestamp']}/lr", self.optimizer.param_groups[0]['lr'], i)

            print_kwarg = [i_str]
            for phase in phases:
                print_kwarg += [self.config[f'{phase}_loss'], self.config[f'{phase}_metric']]
            print_kwarg += [self.optimizer.param_groups[0]['lr'], elapsed_time]

            print_row(kwarg_list=print_kwarg, pad=' ')
            print_row(kwarg_list=['']*len(kwarg_list), pad='-')
            self.dataframe = self.dataframe.append(dict(zip(kwarg_list, print_kwarg)), ignore_index=True)

            if is_best:
                self.save(self.ckpt)

            if save_every_validation:
                self.save(self.ckpt + f"-{self.config[train_unit]}")

    def _step(self, phase, iterator, only_inference=False):
        if self.config['is_data_dict']:
            batch_dict = next(iterator)
            batch_size = batch_dict[list(batch_dict.keys())[0]].size(0)
            for k, v in batch_dict.items():
                batch_dict[k] = v.to(self.device)

        else:
            batch_x, batch_y = next(iterator)

            if isinstance(batch_x, list):
                batch_x = [x.to(self.device) for x in batch_x]
            else:
                batch_x = [batch_x.to(self.device)]

            if isinstance(batch_y, list):
                batch_y = [y.to(self.device) for y in batch_y]
            else:
                batch_y = [batch_y.to(self.device)]

            batch_size = batch_x[0].size(0)

        self.optimizer.zero_grad()
        with torch.set_grad_enabled(phase == "train"):

            if self.config['is_data_dict']:
                outputs = self.net(batch_dict)
                if not only_inference:
                    loss = self.criterion(outputs, batch_dict)
            else:
                outputs = self.net(*batch_x)
                if not only_inference:
                    loss = self.criterion(outputs, *batch_y)

            if only_inference:
                return outputs

            if phase == "train":
                loss.backward()
                if self.config['clip_gradient_norm']:
                    clip_grad_norm_(self.net.parameters(), self.config['clip_gradient_norm'])
                self.optimizer.step()

        with torch.no_grad():
            if self.config['is_data_dict']:
                metric = self.metric(outputs, batch_dict)
            else:
                metric = self.metric(outputs, *batch_y)

        return {'loss': loss.item(),
                'batch_size': batch_size,
                'metric': metric.item()}


    def _train(self, phase, num_steps=0):

        running_loss = AverageMeter()
        running_metric = AverageMeter()

        if phase == 'train':
            self.net.train()
        else: # phase == "val":
            self.net.eval()

        dataloader = self.dataloader[phase]
        step_iterator = iter(dataloader)
        for st in tqdm(range(num_steps), leave=False):
            if (st + 1) % len(dataloader) == 0:
                step_iterator = iter(dataloader)
            results = self._step(phase=phase, iterator=step_iterator)
            running_loss.update(results['loss'], results['batch_size'])
            running_metric.update(results['metric'], results['batch_size'])

        return running_loss.avg, running_metric.avg

    def eval(self, dataloader=None):
        self.net.eval()
        if dataloader is None:
            dataloader = self.dataloader['val']
            phase = "val"

        output_list = []
        step_iterator = iter(dataloader)
        num_steps = len(dataloader)
        for st in tqdm(range(num_steps), leave=False):
            results = self._step(phase="val", iterator=step_iterator, only_inference=True)
            output_list.append(results)

        output_cat = torch.cat(output_list)
        return output_cat

    def swa_apply(self, bn_update=True):
        assert hasattr(self.optimizer, "swap_swa_sgd")
        self.optimizer.swap_swa_sgd()
        if bn_update:
            self.swa_bn_update()

    def swa_bn_update(self):
        r"""Updates BatchNorm running_mean, running_var buffers in the model.

        It performs one pass over data in `loader` to estimate the activation
        statistics for BatchNorm layers in the model.

        original source is from : torchcontrib
        """
        if not check_bn(self.net):
            return
        was_training = self.net.training
        self.net.train()
        momenta = {}
        self.net.apply(reset_bn)
        self.net.apply(lambda module: get_momenta(module, momenta))
        n = 0
        for input in self.dataloader['train']:
            if isinstance(input, (list, tuple)):
                input = input[0]
                b = input.size(0)
                input = input.to(self.device)
            elif self.config['is_data_dict']:
                b = input[list(input.keys())[0]].size(0)
                for k, v in input.items():
                    input[k] = v.to(self.device)
            else:
                b = input.size(0)
                input = input.to(self.device)

            momentum = b / float(n + b)
            for module in momenta.keys():
                module.momentum = momentum

            self.net(input)
            n += b

        self.net.apply(lambda module: set_momenta(module, momenta))
        self.net.train(was_training)

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
    from PinkBlack.trainer import *

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
    trainer.train(10)
