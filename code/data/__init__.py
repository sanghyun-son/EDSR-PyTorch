from importlib import import_module
import dataloader

class data:
    def __init__(self, args):
        self.trainModule = import_module('data.' + args.trainData)
        self.testModule = [(
            import_module('data.' + d), d) for d in args.testData]
        self.args = args

    def getLoader(self):
        trainSet = getattr(self.trainModule, self.args.trainData)(self.args)
        trainLoader = dataloader.MSDataLoader(
            self.args, trainSet, batch_size=self.args.batchSize,
            shuffle=True, pin_memory=True)

        testSet = []
        for m in self.testModule:
            if m[1] == 'benchmark':
                benchmarkList = ['Set5', 'Set14', 'B100', 'Urban100']
                for b in benchmarkList:
                    testSet.append(getattr(m[0], m[1])(self.args, b))
            else:
                testSet.append(getattr(m[0], m[1])(self.args, train=False))

        testLoader = [dataloader.MSDataLoader(
            self.args, s, batch_size=1,
            shuffle=False, pin_memory=True) for s in testSet]

        return (trainLoader, testLoader)
