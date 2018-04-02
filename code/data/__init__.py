from importlib import import_module

from dataloader import MSDataLoader
from  torch.utils.data.dataloader import default_collate

class data:
    def __init__(self):
        pass

    def get_loader(self, args):
        module_train = import_module('data.' + args.data_train.lower())
        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
            module_test = import_module('data.benchmark')
            benchmark = True
        else:
            module_test = import_module('data.' +  args.data_test.lower())
            benchmark = False

        kwargs = {}
        if args.no_cuda:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True

        loader_train = None
        if not args.test_only:
            trainset = getattr(module_train, args.data_train)(args)
            loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs
            )

        if benchmark:
            testset = getattr(module_test, 'Benchmark')(args, train=False)
        else:
            testset = getattr(
                module_test, args.data_test
            )(args, train=False)

        loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            **kwargs
        )

        return loader_train, loader_test

