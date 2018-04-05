import utility
from model import common
from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.args = args
        self.gan_type = gan_type
        self.discriminator = discriminator.Discriminator(args)
        self.optimizer = utility.make_optimizer(args, self.discriminator)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.loss_d = 0

    def forward(self, fake, real):
        self.optimizer.zero_grad()

        d_fake = self.discriminator(fake.detach())
        d_real = self.discriminator(real)

        label_fake = Variable(
            d_fake.data.new(d_fake.size()).fill_(0)
        )
        label_real = Variable(
            d_real.data.new(d_real.size()).fill_(1)
        )

        if self.gan_type == 'GAN':
            loss_d \
                = F.binary_cross_entropy_with_logits(d_fake, label_fake) \
                + F.binary_cross_entropy_with_logits(d_real, label_real)
        elif self.gan_type.find('WGAN') >= 0:
            loss_d = d_fake.mean() - d_real.mean()

        # Discriminator update
        self.loss = loss_d.data[0]
        loss_d.backward()
        self.optimizer.step()

        if self.gan_type == 'WGAN':
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.args.wclip, self.args.wclip)

        d_fake_for_g = self.discriminator(fake)
        if self.gan_type == 'GAN':
            loss_g = F.binary_cross_entropy_with_logits(
                d_fake_for_g, label_real
            )
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_for_g.mean()

        # Loss for the generator
        return loss_g
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_discriminator = self.discriminator.state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars
        )
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)
               
# Some references
# https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# OR
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
