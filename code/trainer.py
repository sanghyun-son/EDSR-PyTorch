import math
import random
from decimal import Decimal
from functools import reduce

import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import torchvision.utils as tu

class Trainer():
    def __init__(self, loader, ckp, args):
        self.args = args
        self.scale = args.scale

        self.loader_train, self.loader_test = loader
        self.model, self.loss, self.optimizer, self.scheduler = ckp.load()
        self.ckp = ckp

        self.log_training = 0
        self.log_test = 0

    def _scale_change(self, idx_scale, testset=None):
        if len(self.scale) > 1:
            if self.args.n_GPUs == 1:
                self.model.set_scale(idx_scale)
            else:
                self.model.module.set_scale(idx_scale)

            if testset is not None:
                testset.dataset.set_scale(idx_scale)

    def train(self):
        self.scheduler.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.ckp.add_log(torch.zeros(1, len(self.loss)))
        self.model.train()

        timer_data, timer_model = utils.timer(), utils.timer()
        for batch, (input, target, idx_scale) in enumerate(self.loader_train):
            input, target = self._prepare(input, target)
            self._scale_change(idx_scale)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self._calc_loss(output, target)
            loss.backward()
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self._display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.ckp.log_training[-1, :] /= len(self.loader_train)

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)), False)
        self.model.eval()

        # We can use custom forward function 
        def _test_forward(x, scale):
            if self.args.self_ensemble:
                return utils.x8_forward(x, self.model, self.args.precision)
            elif self.args.chop_forward:
                return utils.chop_forward(x, self.model, scale)
            else:
                return self.model(x)

        timer_test = utils.timer()
        set_name = type(self.loader_test.dataset).__name__
        for idx_scale, scale in enumerate(self.scale):
            eval_acc = 0
            self._scale_change(idx_scale, self.loader_test)
            for idx_img, (input, target, _) in enumerate(self.loader_test):
                input, target = self._prepare(input, target, volatile=True)
                output = _test_forward(input, scale)
                eval_acc += utils.calc_PSNR(
                    output, target, set_name, self.args.rgb_range, scale)
                self.ckp.save_results(idx_img, input, output, target, scale)

            self.ckp.log_test[-1, idx_scale] = eval_acc / len(self.loader_test)
            best = self.ckp.log_test.max(0)
            performance = 'PSNR: {:.3f}'.format(
                self.ckp.log_test[-1, idx_scale])
            self.ckp.write_log(
                '[{} x{}]\t{} (Best: {:.3f} from epoch {})'.format(
                    set_name,
                    scale,
                    performance,
                    best[0][idx_scale],
                    best[1][idx_scale] + 1))

        if best[1][0] + 1 == epoch:
            is_best = True
        else:
            is_best = False

        self.ckp.write_log(
            'Time: {:.2f}s\n'.format(timer_test.toc()), refresh=True)
        self.ckp.save(self, epoch, is_best=is_best)

    def _prepare(self, input, target, volatile=False):
        if not self.args.no_cuda:
            input = input.cuda()
            target = target.cuda()

        input = Variable(input, volatile=volatile)
        target = Variable(target)
           
        return input, target

    def _calc_loss(self, output, target):
        loss_list = [] 
        
        for i, l in enumerate(self.loss):
            if isinstance(output, list):
                if isinstance(target, list):
                    loss = l['function'](output[i], target[i])
                else:
                    loss = l['function'](output[i], target)
            else:
                loss = l['function'](output, target)

            loss_list.append(l['weight'] * loss)
            self.ckp.log_training[-1, i] += loss.data[0]

        loss_total = reduce((lambda x, y: x + y), loss_list)
        if len(self.loss) > 1:
            self.ckp.log_training[-1, -1] += loss_total.data[0]

        return loss_total

    def _display_loss(self, batch):
        log = [
            '[{}: {:.4f}] '.format(t['type'], l / (batch + 1)) \
            for l, t in zip(self.ckp.log_training[-1], self.loss)]

        return ''.join(log)

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.args.epochs

