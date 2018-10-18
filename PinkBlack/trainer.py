import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

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


class Trainer:
    def __init__(self, net,
                 criterion=None,
                 metric=None,
                 dataloader=None,
                 optimizer=None,
                 lr_scheduler=None,
                 logpath="./log"):
        assert isinstance(dataloader, dict)

        self.net = net
        self.criterion = criterion
        self.metric = metric
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = torch.device("cpu")
        for param in self.net.parameters():
            self.device = param.device
            break

        try:
            from tensorboardX import SummaryWriter
            self.logger = SummaryWriter(logpath)
        except ImportError:
            self.logger = None

    def train(self, epoch):

        from time import time
        from tqdm import tqdm
        for ep in range(1, epoch+1):
            start_time = time()
            train_loss = AverageMeter()
            val_loss = AverageMeter()

            for phase in ['train', 'val']:
                running_loss = AverageMeter()
                running_metric = AverageMeter()

                if phase == 'train':
                    self.net.train()
                    dataloader = self.dataloader['train']
                elif phase == "val":
                    self.net.eval()
                    dataloader = self.dataloader['val']

                for batch_x, batch_y in tqdm(dataloader):
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
                    train_loss = running_loss
                    train_metric = running_metric
                elif phase == "val":
                    val_loss = running_loss
                    val_metric = running_metric

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            elapsed_time = time() - start_time
            self.logger.add_scalars("curves/loss", {'train' : train_loss.avg,
                                                    'val': val_loss.avg}, ep)
            self.logger.add_scalars("curves/metric", {'train' : train_metric.avg,
                                                    'val': val_metric.avg}, ep)
            self.logger.add_scalar("curves/time", elapsed_time, ep)
            # self.log(train_loss.avg, "train_loss", epoch)
            # self.log(train_metric.avg, "train_metric", epoch)
            # self.log(val_loss.avg, "val_loss", epoch)
            # self.log(val_metric.avg, "val_metric", epoch)


        pass

    def log(self, msg, step):
        pass

if __name__ == "__main__":
    # demo.py

    import torch
    import torchvision.utils as vutils
    import numpy as np
    import torchvision.models as models
    from torchvision import datasets
    from tensorboardX import SummaryWriter

    resnet18 = models.resnet18(False)
    writer = SummaryWriter()
    sample_rate = 44100
    freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

    for n_iter in range(100):

        dummy_s1 = torch.rand(1)
        dummy_s2 = torch.rand(1)
        # data grouping by `slash`
        writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
        writer.add_scalar('data/scalar2', dummy_s2[0], n_iter)

        writer.add_scalars('data/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
                                                 'xcosx': n_iter * np.cos(n_iter),
                                                 'arctanx': np.arctan(n_iter)}, n_iter)

        dummy_img = torch.rand(32, 3, 64, 64)  # output from network
        if n_iter % 10 == 0:
            x = vutils.make_grid(dummy_img, normalize=True, scale_each=True)
            writer.add_image('Image', x, n_iter)

            dummy_audio = torch.zeros(sample_rate * 2)
            for i in range(x.size(0)):
                # amplitude of sound should in [-1, 1]
                dummy_audio[i] = np.cos(freqs[n_iter // 10] * np.pi * float(i) / float(sample_rate))
            writer.add_audio('myAudio', dummy_audio, n_iter, sample_rate=sample_rate)

            writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)

            for name, param in resnet18.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)

            # needs tensorboard 0.4RC or later
            writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), n_iter)

    dataset = datasets.MNIST('mnist', train=False, download=True)
    images = dataset.test_data[:100].float()
    label = dataset.test_labels[:100]

    features = images.view(100, 784)
    writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
