import torch

import utility
from option import args
from data import data
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    my_loader = data(args).get_loader()
    t = Trainer(my_loader, checkpoint, args)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

