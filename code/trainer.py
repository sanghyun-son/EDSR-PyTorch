import math
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable

class Trainer():
    def __init__(self, loader, ckp, args):
        self.args = args
        self.scale = args.scale

        self.loader_train, self.loader_test = loader
        self.model, self.loss, self.optimizer, self.scheduler = ckp.load()
        self.ckp = ckp

        self.log_training = 0
        self.log_test = 0
        self.error_last = 1e8

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

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            self._scale_change(idx_scale)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self._calc_loss(sr, hr)
            if loss.data[0] < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.data[0]
                ))

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
        self.error_last = self.ckp.log_training[-1, :][0]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)), False)
        self.model.eval()

        # We can use custom forward function 
        def _test_forward(x, scale):
            if self.args.self_ensemble:
                return utility.x8_forward(x, self.model, self.args.precision)
            elif self.args.chop_forward:
                return utility.chop_forward(x, self.model, scale)
            else:
                return self.model(x)

        timer_test = utility.timer()
        set_name = self.args.data_test
        for idx_scale, scale in enumerate(self.scale):
            eval_acc = 0
            self._scale_change(idx_scale, self.loader_test)
            for idx_img, (lr, hr, _) in enumerate(self.loader_test):
                no_eval = isinstance(hr[0], torch._six.string_classes)
                if no_eval:
                    lr = self.prepare([lr], volatile=True)[0]
                    filename = hr[0]
                else:
                    lr, hr = self.prepare([lr, hr], volatile=True)
                    filename = idx_img + 1

                rgb_range = self.args.rgb_range
                sr = _test_forward(lr, scale)
                sr = utility.quantize(sr, rgb_range)

                if no_eval:
                    save_list = [sr]
                else:
                    eval_acc += utility.calc_PSNR(
                        sr,
                        hr.div(rgb_range),
                        set_name,
                        scale
                    )
                    save_list = [sr, lr.div(rgb_range), hr.div(rgb_range)]

                if self.args.save_results:
                    self.ckp.save_results(filename, save_list, scale)

            self.ckp.log_test[-1, idx_scale] = eval_acc / len(self.loader_test)
            best = self.ckp.log_test.max(0)
            performance = 'PSNR: {:.3f}'.format(
                self.ckp.log_test[-1, idx_scale]
            )
            self.ckp.write_log(
                '[{} x{}]\t{} (Best: {:.3f} from epoch {})'.format(
                    set_name,
                    scale,
                    performance,
                    best[0][idx_scale],
                    best[1][idx_scale] + 1
                )
            )

        is_best = (best[1][0] + 1 == epoch)
        self.ckp.write_log(
            'Time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        self.ckp.save(self, epoch, is_best=is_best)

    def prepare(self, l, volatile=False):
        def _prepare(idx, tensor):
            if not self.args.no_cuda:
                tensor = tensor.cuda()

            if self.args.precision == 'half':
                tensor = tensor.half()

            # Only test lr can be volatile
            var = Variable(tensor, volatile=(volatile and idx==0))
            
            return var
           
        return [_prepare(i, _l) for i, _l in enumerate(l)]

    def _calc_loss(self, sr, hr):
        loss_list = [] 
        
        for i, l in enumerate(self.loss):
            if isinstance(sr, list):
                if isinstance(hr, list):
                    loss = l['function'](sr[i], hr[i])
                else:
                    loss = l['function'](sr[i], hr)
            else:
                loss = l['function'](sr, hr)

            loss_list.append(l['weight'] * loss)
            self.ckp.log_training[-1, i] += loss.data[0]

        loss_total = sum(loss_list)
        if len(self.loss) > 1:
            self.ckp.log_training[-1, -1] += loss_total.data[0]

        return loss_total

    def _display_loss(self, batch):
        n_samples = batch + 1
        log = [
            '[{}: {:.4f}] '.format(t['type'], l / n_samples) \
            for l, t in zip(self.ckp.log_training[-1], self.loss)]

        return ''.join(log)

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
