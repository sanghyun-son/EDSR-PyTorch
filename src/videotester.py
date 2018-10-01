import os
import math

import utility
from data import common

import torch
import cv2

from tqdm import tqdm

class VideoTester():
    def __init__(self, args, my_model, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.model = my_model

        self.filename, _ = os.path.splitext(os.path.basename(args.dir_demo))

    def test(self):
        self.ckp.write_log('\nEvaluation on video:')
        self.model.eval()

        timer_test = utility.timer()
        torch.set_grad_enabled(False)
        for idx_scale, scale in enumerate(self.scale):
            vidcap = cv2.VideoCapture(self.args.dir_demo)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_batches = math.ceil(total_frames / self.args.batch_size)
            vidwri = cv2.VideoWriter(
                self.ckp.get_path('{}_x{}.avi'.format(self.filename, scale)),
                cv2.VideoWriter_fourcc(*'XVID'),
                int(vidcap.get(cv2.CAP_PROP_FPS)),
                (
                    int(scale * vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(scale * vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
            )

            tqdm_test = tqdm(range(total_batches), ncols=80)
            for _ in tqdm_test:
                fs = []
                for _ in range(self.args.batch_size):
                    success, lr = vidcap.read()
                    if success:
                        fs.append(lr)
                    else:
                        break

                fs = common.set_channel(*fs, n_channels=self.args.n_colors)
                fs = common.np2Tensor(*fs, rgb_range=self.args.rgb_range)
                lr = torch.stack(fs, dim=0)
                lr, = self.prepare(lr)
                sr = self.model(lr, idx_scale)
                sr = utility.quantize(sr, self.args.rgb_range)

                for i in range(self.args.batch_size):
                    normalized = sr[i].mul(255 / self.args.rgb_range)
                    ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                    vidwri.write(ndarr)

            self.vidcap.release()
            self.vidwri.release()

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

