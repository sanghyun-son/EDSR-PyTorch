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
        torch.set_grad_enabled(False)

        self.ckp.write_log('\nEvaluation on video:')
        self.model.eval()

        timer_test = utility.timer()
        for idx_scale, scale in enumerate(self.scale):
            vidcap = cv2.VideoCapture(self.args.dir_demo)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            vidwri = cv2.VideoWriter(
                self.ckp.get_path('{}_x{}.avi'.format(self.filename, scale)),
                cv2.VideoWriter_fourcc(*'XVID'),
                vidcap.get(cv2.CAP_PROP_FPS),
                (
                    int(scale * vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(scale * vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
            )

            tqdm_test = tqdm(range(total_frames), ncols=80)
            for _ in tqdm_test:
                success, lr = vidcap.read()
                if not success: break

                lr, = common.set_channel(lr, n_channels=self.args.n_colors)
                lr, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)
                lr, = self.prepare(lr.unsqueeze(0))
                sr = self.model(lr, idx_scale)
                sr = utility.quantize(sr, self.args.rgb_range).squeeze(0)

                normalized = sr * 255 / self.args.rgb_range
                ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                vidwri.write(ndarr)

            vidcap.release()
            vidwri.release()

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

