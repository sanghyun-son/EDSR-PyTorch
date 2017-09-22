import os
import math
import time
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import skimage.io as sio
import skimage.color as sc

from model import model
from loss import loss

import torch
import torch.optim as optim
import torchvision.utils as tUtils
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
        self.ok = True
        now = datetime.datetime.now()

        if args.load == '.':
            if args.save == '.':
                args.save = now.strftime('%Y-%m-%d-%H:%M:%S')
            self.dir = '../experiment/' + args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'

        if args.saveModel > -1:
            self.saveModel(args.saveModel)
            self.ok = False

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _makeDirs(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _makeDirs(self.dir)
        _makeDirs(self.dir + '/model')
        _makeDirs(self.dir + '/results')

        if os.path.exists(self.dir + '/log.txt'):
            self.logFile = open(self.dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.dir + '/log.txt', 'w')

        self.args = args

    def load(self):
        myModel = model(self.args).getModel()
        trainable = filter(lambda x: x.requires_grad, myModel.parameters())

        if self.args.optimizer == 'SGD':
            myOptimizer = optim.SGD(
                trainable,
                lr=self.args.lr,
                momentum=self.args.momentum)
        elif self.args.optimizer == 'ADAM':
            myOptimizer = optim.Adam(
                trainable,
                lr=self.args.lr,
                betas=(self.args.beta1, self.args.beta2),
                eps=self.args.epsilon)
        elif self.args.optimizer == 'RMSprop':
            myOptimizer = optim.RMSprop(
                trainable,
                lr=self.args.lr,
                eps=self.args.epsilon)
        if self.args.load == '.':
            myLoss = loss(self.args).getLoss()
            self.trainingLog = torch.Tensor()
            self.testLog = torch.Tensor()
        else:
            myModel.load_state_dict(
                torch.load(self.dir + '/model/model_lastest.pt'))
            myLoss = torch.load(self.dir + '/loss.pt')
            myOptimizer.load_state_dict(
                torch.load(self.dir + '/optimizer.pt'))
            self.trainingLog = torch.load(self.dir + '/trainingLog.pt')
            self.testLog = torch.load(self.dir + '/testLog.pt')
            print('Load loss function from checkpoint...')
            print('Continue from epoch {}...'.format(len(self.testLog)))

        return myModel, myLoss, myOptimizer

    def addLog(self, log, train=True):
        if train:
            self.trainingLog = torch.cat([self.trainingLog, log])
        else:
            self.testLog = torch.cat([self.testLog, log])

    def save(self, trainer, epoch):
        torch.save(
            trainer.model.state_dict(),
            self.dir + '/model/model_lastest.pt')
        torch.save(
            trainer.model,
            self.dir + '/model/model_obj.pt')
        if not self.args.testOnly:
            torch.save(
                trainer.model.state_dict(),
                '{}/model/model_{}.pt'.format(self.dir, epoch))
            torch.save(
                trainer.loss,
                self.dir + '/loss.pt')
            torch.save(
                trainer.optimizer.state_dict(),
                self.dir + '/optimizer.pt')
            torch.save(
                self.trainingLog,
                self.dir + '/trainingLog.pt')
            torch.save(
                self.testLog,
                self.dir + '/testLog.pt')
            self.plot(trainer, epoch, self.trainingLog, self.testLog, self.dir)

    def saveLog(self, log, refresh=False):
        print(log)
        self.logFile.write(log + '\n')
        if refresh:
            self.logFile.close()
            self.logFile = open(self.dir + '/log.txt', 'a')

    def saveModel(self, epoch):
        if not self.args.testOnly:
            print('Save the model for evaluation...')
            if epoch > 0:
                modelPath = '{}/model/model_{}.pt'.format(self.dir, epoch)
            else:
                modelPath = '{}/model/model_lastest.pt'.format(self.dir)

            modelHolder = model(self.args).getModel()
            modelHolder.load_state_dict(torch.load(modelPath))
            torch.save(modelHolder, '../demo/model/{}.pt'.format(self.args.save))

    def done(self):
        self.logFile.close()
        self.saveModel(0)

    def plot(self, trainer, epoch, training, test, dir):
        axis = np.linspace(1, epoch, epoch)

        for i, loss in enumerate(trainer.loss):
            fig = plt.figure()
            label = '{} Loss'.format(loss['type'])
            plt.title(label)
            plt.xlabel('Epochs')
            plt.grid(True)
            plt.plot(axis, training[:, i].numpy(), label=label)
            plt.legend()
            plt.savefig('{}/loss_{}.pdf'.format(dir, loss['type']))
            plt.close(fig)

        setName = trainer.testLoader.dataset.name
        fig = plt.figure()
        label = 'SR on {}'.format(setName)
        plt.title(label)
        plt.xlabel('Epochs')
        plt.grid(True)
        for scaleIdx, scale in enumerate(self.args.scale):
            legend = 'Scale {}'.format(scale)
            plt.plot(
                axis,
                test[:, scaleIdx].numpy(),
                label=legend)
            plt.legend()

        plt.savefig(
            '{}/test_SR_{}.pdf'.format(dir, setName))
        plt.close(fig)

    def getEpoch(self):
        return len(self.testLog) + 1

    def saveResults(self, idx, input, output, target, scale):
        idx += 1
        if self.args.saveResults:
            fileName = '{}/results/{}x{}_'.format(
                self.dir, idx, scale)
            tUtils.save_image(
                input.data[0] / self.args.rgbRange, fileName + 'LR.png')
            tUtils.save_image(
                output.data[0] / self.args.rgbRange, fileName + 'SR.png')
            if target is not None:
                tUtils.save_image(
                    target.data[0] / self.args.rgbRange,
                    fileName + 'GT.png')
            
def x8Forward(img, model, precision='single'):
    inputList = []
    outputList = []
    n = 8

    def _transform(v, op):
        if precision == 'half':
            v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        
        ret = torch.Tensor(tfnp).cuda()

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        return Variable(ret, volatile=True)

    for i in range(n):
        if i == 0:
            inputList.append(img)
        elif i == 1:
            inputList.append(_transform(img, 'vflip'))
        elif i > 1 and i <= 3:
            inputList.append(_transform(inputList[i - 2], 'hflip'))
        elif i > 3:
            inputList.append(_transform(inputList[i - 4], 'transpose'))

        outputList.append(model(inputList[-1]))

    for i in range(n - 1, -1, -1):
        if i > 3:
            outputList[i] = _transform(outputList[i], 'transpose')
        if i % 4 > 1:
            outputList[i] = _transform(outputList[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputList[i] = _transform(outputList[i], 'vflip')

        if i != 0:
            outputList[0] += outputList[i]

    outputList[0] /= n

    return outputList[0]

def quantize(img, rgbRange):
    return img.mul(255 / rgbRange).clamp(0, 255).add(0.5).floor().div(255)

def rgb2ycbcrT(rgb):
    rgb = rgb.numpy().transpose(1, 2, 0)
    yCbCr = sc.rgb2ycbcr(rgb) / 255

    return torch.Tensor(yCbCr[:, :, 0])

def calcPSNR(input, target, setName, rgbRange, scale):
    # Do not calculate PSNR for user input
    if target is None:
        return 0

    # We will evaluate these datasets in y channel only
    yList = ['Set5', 'Set14', 'B100', 'Urban100']

    (_, c, h, w) = input.size()
    input = quantize(input.data[0], rgbRange)
    target = quantize(target[:, :, 0:h, 0:w].data[0], rgbRange)
    diff = input - target
    if setName in yList:
        shave = scale
        if c > 1:
            inputY = rgb2ycbcrT(input.cpu())
            targetY = rgb2ycbcrT(target.cpu())
            diff = (inputY - targetY).view(1, h, w)
    else:
        shave = scale + 6

    diffShave = diff[:, shave:(h - shave), shave:(w - shave)]
    mse = diffShave.pow(2).mean()
    psnr = -10 * np.log10(mse)

    return psnr

