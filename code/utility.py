import os
import math
import time
import datetime
from functools import reduce

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import skimage.io as sio
import skimage.color as sc

import loss
import model

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.utils as tu
from torch.autograd import Variable

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = '../experiment/' + args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir): args.load = '.'

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def load(self):
        my_model = model.Model(self.args)
        trainable = filter(lambda x: x.requires_grad, my_model.parameters())

        if self.args.optimizer == 'SGD':
            optimizer_function = optim.SGD
            kwargs = {'momentum': self.args.momentum}
        elif self.args.optimizer == 'ADAM':
            optimizer_function = optim.Adam
            kwargs = {
                'betas': (self.args.beta1, self.args.beta2),
                'eps': self.args.epsilon
            }
        elif self.args.optimizer == 'RMSprop':
            optimizer_function = optim.RMSprop
            kwargs = {'eps': self.args.epsilon}

        kwargs['lr'] = self.args.lr
        kwargs['weight_decay'] = 0
        my_optimizer = optimizer_function(trainable, **kwargs)

        if self.args.decay_type == 'step':
            my_scheduler = lrs.StepLR(
                my_optimizer,
                step_size=self.args.lr_decay,
                gamma=self.args.gamma)

        elif self.args.decay_type.find('step') >= 0:
            milestones = self.args.decay_type.split('_')
            milestones.pop(0)
            milestones = list(map(lambda x: int(x), milestones))
            my_scheduler = lrs.MultiStepLR(
                my_optimizer,
                milestones=milestones,
                gamma=self.args.gamma)

        self.log_training = torch.Tensor()
        self.log_test = torch.Tensor()
        my_loss = loss.Loss(self.args)
        if self.args.load != '.':
            if not self.args.test_only:
                self.log_training = torch.load(self.dir + '/log_training.pt')
                self.log_test = torch.load(self.dir + '/log_test.pt')

            resume = self.args.resume
            if resume == -1:
                my_model.load_state_dict(
                    torch.load(self.dir + '/model/model_latest.pt')
                )
                resume = len(self.log_test)
            else:
                my_model.load_state_dict(
                    torch.load(self.dir + '/model/model_{}.pt'.format(resume))
                )

            print('Load loss function from checkpoint...')
            my_loss.load_state_dict(torch.load(self.dir + '/loss.pt'))
            my_optimizer.load_state_dict(
                torch.load(self.dir + '/optimizer.pt')
            )

            print('Continue from epoch {}...'.format(resume))

        return my_model, my_loss, my_optimizer, my_scheduler

    def add_log(self, log, train=True):
        if train:
            self.log_training = torch.cat([self.log_training, log])
        else:
            self.log_test = torch.cat([self.log_test, log])

    def save(self, trainer, epoch, is_best=False):
        state = trainer.model.state_dict()

        save_list = [(state, 'model/model_latest.pt')]
        if not self.args.test_only:
            if is_best:
                save_list.append((state, 'model/model_best.pt'))
            if self.args.save_models:
                save_list.append((state, 'model/model_{}.pt'.format(epoch)))

            save_list.append((trainer.loss.state_dict(), 'loss.pt'))
            save_list.append((trainer.optimizer.state_dict(), 'optimizer.pt'))
            save_list.append((self.log_training, 'log_training.pt'))
            save_list.append((self.log_test, 'log_test.pt'))
            self.plot(trainer, epoch, self.log_training, self.log_test)

        for o, p in save_list:
            torch.save(o, os.path.join(self.dir, p))

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot(self, trainer, epoch, training, test):
        axis = np.linspace(1, epoch, epoch)

        def _init_figure(label):
            fig = plt.figure()
            plt.title(label)
            plt.xlabel('Epochs')
            plt.grid(True)

            return fig
           
        def _close_figure(fig, filename):
            plt.savefig(filename)
            plt.close(fig)

        for i, loss in enumerate(trainer.loss):
            label = '{} Loss'.format(loss['type'])
            fig = _init_figure(label)
            plt.plot(axis, training[:, i].numpy(), label=label)
            plt.legend()
            _close_figure(fig, '{}/loss_{}.pdf'.format(self.dir, loss['type']))

        set_name = type(trainer.loader_test.dataset).__name__
        fig = _init_figure('SR on {}'.format(set_name))
        for idx_scale, scale in enumerate(self.args.scale):
            legend = 'Scale {}'.format(scale)
            plt.plot(axis, test[:, idx_scale].numpy(), label=legend)
            plt.legend()

        _close_figure(
            fig,
            '{}/test_{}.pdf'.format(self.dir, set_name))

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            tu.save_image(
                v.data[0],
                '{}{}.png'.format(filename, p),
                padding=0
            )

def quantize(img, rgb_range):
    return img.mul(255 / rgb_range).clamp(0, 255).round()

def calc_PSNR(sr, hr, scale, benchmark=False):
    '''
        Here we assume quantized(0-255) arguments.
        For Set5, Set14, B100, Urban100 dataset,
        we measure PSNR on luminance channel only
    '''
    diff = (sr - hr).data.div(255)
    _, c, h, w = diff.size()

    # We will evaluate these datasets in y channel only
    if benchmark:
        shave = scale
        if c > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6

    valid = diff[:, :, shave:(h-shave), shave:(w-shave)]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)
