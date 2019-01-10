import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import shutil

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


def padding(arg, width, pad=' '):
    if isinstance(arg, float):
        return '{:.6f}'.format(arg).center(width, pad)
    elif isinstance(arg, int):
        return '{:6d}'.format(arg).center(width, pad)
    elif isinstance(arg, str):
        return arg.center(width, pad)
    elif isinstance(arg, tuple):
        if len(arg) != 2:
            raise ValueError('Unknown type: {}'.format(type(arg), arg))
        if not isinstance(arg[1], str):
            raise ValueError('Unknown type: {}'
                             .format(type(arg[1]), arg[1]))
        return padding(arg[0], width, pad=pad)
    else:
        raise ValueError('Unknown type: {}'.format(type(arg), arg))


def print_row(kwarg_list=[], pad=' '):
    len_kwargs = len(kwarg_list)
    term_width = shutil.get_terminal_size().columns
    width = min((term_width - 1 - len_kwargs) * 9 // 10, 150) // len_kwargs
    row = '|{}' * len_kwargs + '|'
    columns = []
    for kwarg in kwarg_list:
        columns.append(padding(kwarg, width, pad=pad))
    print(row.format(*columns))


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
                 logdir="./log",
                 ckpt="./ckpt/ckpt.pth"):

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
        self.logdir = logdir
        self.ckpt = ckpt

        self.device = torch.device("cpu")
        for param in self.net.parameters():
            self.device = param.device
            break

        if self.logdir is not None:
            try:
                from tensorboardX import SummaryWriter
                self.logger = SummaryWriter(self.logdir)
            except ImportError:
                self.logger = None
        else:
            self.logger = None


    def train(self, epoch):

        from time import time
        from tqdm import tqdm
        from datetime import datetime
        kwarg_list = ['epoch', 'train_loss', 'train_metric',
                      'val loss', 'val metric', 'time']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print_row(kwarg_list=['']*len(kwarg_list), pad='-')
        print_row(kwarg_list=kwarg_list, pad=' ')
        print_row(kwarg_list=['']*len(kwarg_list), pad='-')

        min_val_loss = 1e8
        max_val_metric = -1e8
        for ep in range(1, epoch+1):
            start_time = time()
            train_loss = AverageMeter()
            val_loss = AverageMeter()
            train_metric = AverageMeter()
            val_metric = AverageMeter()

            for phase in ['train', 'val']:

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

                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.net(batch_x)
                        loss = self.criterion(outputs, batch_y)
                        metric = self.metric(outputs, batch_y)

                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    running_loss.update(loss.item(), batch_x.size(0))
                    running_metric.update(metric.item(), batch_x.size(0))

                if phase == "train":
                    train_loss = running_loss.avg
                    train_metric = running_metric.avg
                elif phase == "val":
                    val_loss = running_loss.avg
                    val_metric = running_metric.avg

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            ep_str = str(ep)
            if max_val_metric < val_metric:
                max_val_metric = val_metric
                self.save()
                ep_str = (str(ep)) + '-best'

            elapsed_time = time() - start_time
            if self.logger is not None:
                self.logger.add_scalars(f"{timestamp}/loss", {'train' : train_loss,
                                                        'val': val_loss}, ep)
                self.logger.add_scalars(f"{timestamp}/metric", {'train' : train_metric,
                                                        'val': val_metric}, ep)
                self.logger.add_scalar(f"{timestamp}/time", elapsed_time, ep)

            print_row(kwarg_list=[ep_str, train_loss, train_metric,
                                  val_loss, val_metric, elapsed_time], pad=' ')
            print_row(kwarg_list=['']*len(kwarg_list), pad='-')

    def save(self):
        os.makedirs(os.path.dirname(self.ckpt), exist_ok=True)
        if hasattr(self.net, 'module'):
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()
        torch.save(state_dict, self.ckpt)


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
    trainer.train(100)
