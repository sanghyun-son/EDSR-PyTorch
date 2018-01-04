from importlib import import_module

import dataloader

class data:
    def __init__(self, args):
        self.args = args

    def get_loader(self):
        self.module_train = import_module('data.' + self.args.data_train)
        self.module_test = import_module('data.' +  self.args.data_test)

        loader_train = None
        if not self.args.test_only:
            trainset = getattr(
                self.module_train, self.args.data_train)(self.args)
            loader_train = dataloader.MSDataLoader(
                self.args,
                trainset,
                batch_size=self.args.batch_size,
                shuffle=True,
                pin_memory=True)

        testset = getattr(self.module_test, self.args.data_test)(
            self.args, train=False)
        loader_test = dataloader.MSDataLoader(
            self.args,
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=True)

        return loader_train, loader_test

