import os
from data import srdata
from data import div2k

class DIV2KJPEG(div2k.DIV2K):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.q_factor = int(name.replace('DIV2K-Q', ''))
        super(DIV2KJPEG, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'DIV2K')
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(
            self.apath, 'DIV2K_Q{}'.format(self.q_factor)
        )
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.jpg')

