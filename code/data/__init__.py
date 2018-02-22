from importlib import import_module

from dataloader import MSDataLoader
from  torch.utils.data.dataloader import default_collate

class data:
    def __init__(self, args):
        self.args = args

    def get_loader(self):
        self.module_train = import_module('data.' + self.args.data_train)
        self.module_test = import_module('data.' +  self.args.data_test)

        kwargs = {}
        if self.args.no_cuda:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True

        loader_train = None
        if not self.args.test_only:
            trainset = getattr(
                self.module_train, self.args.data_train)(self.args)
            loader_train = MSDataLoader(
                self.args,
                trainset,
                batch_size=self.args.batch_size,
                shuffle=True,
                **kwargs)

        testset = getattr(self.module_test, self.args.data_test)(
            self.args, train=False)
        loader_test = MSDataLoader(
            self.args,
            testset,
            batch_size=1,
            shuffle=False,
            **kwargs)

        return loader_train, loader_test

