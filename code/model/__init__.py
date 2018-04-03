from importlib import import_module

import torch
import torch.nn as nn
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop_forward
        self.precision = args.precision
        self.n_GPUs = args.n_GPUs

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args)
        if args.pre_train != '.':
            print('Loading model from {}...'.format(args.pre_train))
            self.model.load_state_dict(torch.load(args.pre_train))

        if not args.no_cuda:
            print('\tCUDA is ready!')
            torch.cuda.manual_seed(args.seed)
            self.model.cuda()

            if args.precision == 'half':
                self.model.half()

            if args.n_GPUs > 1:
                self.model = nn.DataParallel(self.model, range(0, args.n_GPUs))

        if args.print_model:
            print(self.model)

    def forward(self, x, idx_scale):
        target = self.get_target()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)

        if self.self_ensemble and not self.training:
            return self.x8_forward(x)
        elif self.chop_forward and not self.training:
            return self.chop_forward(x, self.scale[idx_scale])
        else:
            return self.model(x)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        target = self.get_target()
        return target.state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars
        )

    def get_target(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def chop_forward(self, x, scale, shave=10, min_size=80000):
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.chop_forward(patch, scale, shave, min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = Variable(x.data.new(b, c, h, w), volatile=True)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def x8_forward(self, x):
        def _transform(v, op):
            if self.precision != 'single':
                v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'vflip':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'hflip':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 'transpose':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()
            
            ret = torch.Tensor(tfnp).cuda()

            if self.precision == 'half':
                ret = ret.half()

            return Variable(ret, volatile=True)

        lr_list = [x]
        for tf in 'vflip', 'hflip', 'transpose':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [self.model(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 'transpose')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'hflip')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'vflip')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)
        #output = output_cat.median(dim=0, keepdim=True)[0]

        return output