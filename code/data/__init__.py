from importlib import import_module
import dataloader

class data:
    def __init__(self, args):
        self.trainModule = import_module('data.' + args.trainData)
        self.testModule = [(
            import_module('data.' + d), d) for d in args.testData]
        self.args = args

    def getLoader(self):
        if not self.args.testOnly:
            trainSet = getattr(self.trainModule, self.args.trainData)(self.args)
            trainLoader = dataloader.MSDataLoader(
                self.args, trainSet, batch_size=self.args.batchSize,
                shuffle=True, pin_memory=True)
        else:
            trainLoader = None

        testSet = []
        for m in self.testModule:
            testSet = getattr(m[0], m[1])(self.args, train=False)

        testLoader = dataloader.MSDataLoader(
            self.args, testSet, batch_size=1,
            shuffle=False, pin_memory=True)

        return (trainLoader, testLoader)
