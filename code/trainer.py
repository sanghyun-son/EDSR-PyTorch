import math
import random
from decimal import Decimal

import utils

import torch
import torch.optim as optim
from torch.autograd import Variable

class trainer():
    def __init__(self, loader, checkpoint, args):
        self.trainLoader, self.testLoader = loader
        self.model, self.loss, self.optimizer = checkpoint.load()
        self.checkpoint = checkpoint
        self.args = args

        self.trainingLog = 0
        self.testLog = 0

        self.scale = args.scale

    def scaleChange(self, scaleIdx, testSet=None):
        if len(self.scale) > 1:
            self.model.setScale(scaleIdx)
            if testSet is not None:
                testSet.dataset.setScale(scaleIdx)

    def train(self):
        self.model.train()
        lr = self.setLr()
        epoch = self.checkpoint.getEpoch()
        self.checkpoint.saveLog(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))

        dataTimer, modelTimer = utils.timer(), utils.timer()

        self.checkpoint.addLog(torch.zeros(1, len(self.loss)))

        for batch, (input, target, scaleIdx) in enumerate(self.trainLoader):
            input, target = self.prepareData(input, target)
            self.scaleChange(scaleIdx)

            dataTimer.hold()
            modelTimer.tic()
            self.optimizer.zero_grad()
            self.calcLoss(self.model(input), target).backward()
            self.optimizer.step()
            modelTimer.hold()

            if (batch + 1) % self.args.printEvery == 0:
                self.checkpoint.saveLog('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batchSize,
                    len(self.trainLoader.dataset),
                    self.showLoss(batch),
                    modelTimer.release(), dataTimer.release()))

            dataTimer.tic()

        self.checkpoint.trainingLog[-1, :] /= len(self.trainLoader)

    def test(self):
        self.model.eval()
        self.checkpoint.saveLog('\nEvaluation:')
        epoch = self.checkpoint.getEpoch()
        scale = self.scale[0]

        testTimer = utils.timer()
        self.checkpoint.addLog(
            torch.zeros(
                1,
                len(self.args.task),
                len(self.testLoader),
                len(self.scale)), False)

        for setIdx, testSet in enumerate(self.testLoader):
            setName = testSet.dataset.name if testSet.dataset.name else 'Test set'
            testTimer.tic()
            for scaleIdx in range(len(self.scale)):
                scale = self.scale[scaleIdx]
                self.scaleChange(scaleIdx, testSet)
                for imgIdx, (input, target, _) in enumerate(testSet):
                    input, target = self.prepareData(input, target, volatile=True)

                    # Self ensemble!
                    if self.args.selfEnsemble:
                        output = utils.x8Forward(
                            input, self.model, self.args.precision)
                    else:
                        output = self.model(input)

                    evalLog = self.evaluate(
                        self.args, input, output, target, locals())

                if len(self.scale) > 1:
                    best = self.checkpoint.testLog.squeeze(0).max(0)
                else:
                    best = self.checkpoint.testLog.max(0)

                for taskIdx, task in enumerate(self.args.task):
                    performance = '{}: {:.3f}'.format(
                        evalLog[taskIdx],
                        self.checkpoint.testLog[-1, taskIdx, setIdx, scaleIdx])
                    self.checkpoint.saveLog(
                        '[{} on {} x{}]\t{} (Best: {:.3f} from epoch {})'.format(
                            task, setName, scale,
                            performance,
                            best[0][taskIdx, setIdx, scaleIdx],
                            best[1][taskIdx, setIdx, scaleIdx] + 1))
            self.checkpoint.saveLog('Time: {:.2f}s'.format(testTimer.toc()))

        self.checkpoint.saveLog('', refresh=True)
        self.checkpoint.save(self, epoch)

    def setLr(self):
        epoch = self.checkpoint.getEpoch()
        lrDecay = self.args.lrDecay
        decayType = self.args.decayType

        if decayType == 'step':
            epochs = (epoch - 1) // lrDecay
            lr = self.args.lr / self.args.decayFactor**epochs
        elif decayType == 'exp':
            k = math.log(2) / lrDecay
            lr = self.args.lr * math.exp(-k * epoch)
        elif decayType == 'inv':
            k = 1 / lrDecay
            lr = self.args.lr / (1 + k * epoch)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def prepareData(self, input, target, volatile=False):
        if self.args.cuda:
            input = Variable(input.cuda(), volatile=volatile)
            target = Variable(target.cuda())

        if self.args.model == 'S2R':
            self.model.setMask(mask)

        if self.args.precision == 'half':
            input = input.half()
            target = target.half()
        elif self.args.precision == 'double':
            input = input.double()
            target = target.double()
            
        return input, target

    def calcLoss(self, output, target):
        check = self.checkpoint
        totalLoss = 0 

        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                if self.args.multiOutput:
                    if self.args.multiTarget:
                        loss = l['function'](output[i], target[i])
                    else:
                        loss = l['function'](output[i], target)
                else:
                    loss = l['function'](output, target)
                totalLoss += l['weight'] * loss
                check.trainingLog[-1, i] += loss.data[0]
        if len(self.loss) > 1:
            check.trainingLog[-1, -1] += totalLoss.data[0]

        return totalLoss

    def showLoss(self, batch):
        lossLog = ''
        for i, lossType in enumerate(self.loss):
            lossLog += '[{}: {:.4f}] '.format(
                lossType['type'], self.checkpoint.trainingLog[-1, i] / batch)

        return lossLog

    def evaluate(self, args, input, output, target, etc):
        def _doEval(output, target, task):
            setName = etc['setName']
            if task == 'SR':
                return utils.calcPSNR(
                    output, target, setName,
                    self.args.rgbRange,
                    etc['scale']), 'PSNR'
            else:
                return 0, 'None'

        evalLog = [None] * len(args.task)
        for taskIdx, task in enumerate(self.args.task):
            outputN = output
            if etc['setName'] == 'myImage':
                targetN = None
            else:
                targetN = target

            evalValue, evalLog[taskIdx] = _doEval(outputN, targetN, task)

            setIdx = etc['setIdx']
            scaleIdx = etc['scaleIdx']
            imgIdx = etc['imgIdx']

            self.checkpoint.testLog[-1, taskIdx, setIdx, scaleIdx] \
                += evalValue / len(etc['testSet'])
            self.checkpoint.saveResults(
                setIdx, imgIdx,
                input, outputN, targetN, task, self.scale[scaleIdx])

        return evalLog

    def terminate(self):
        if self.args.testOnly:
            self.test()
            return True
        else:
            epoch = self.checkpoint.getEpoch()
            if epoch > self.args.epochs:
                return True
            else:
                return False
