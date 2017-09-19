import torch

import utils
from option import args
from data import data
from trainer import trainer

torch.manual_seed(args.seed)
checkpoint = utils.checkpoint(args)

if checkpoint.ok:
    myLoader = data(args).getLoader()
    t = trainer(myLoader, checkpoint, args)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()
