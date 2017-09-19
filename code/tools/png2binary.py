import os
import argparse

import skimage
import skimage.io as sio

import torch

parser = argparse.ArgumentParser(description='Pre-processing DIV2K .png images')

parser.add_argument('--pathFrom', default='../../../../dataset/DIV2K', metavar='DIR',
                    help='directory of images to convert')
parser.add_argument('--pathTo', default='../../../../dataset/DIV2K_decoded', metavar='DIR',
                    help='directory of images to save')
parser.add_argument('--split', default=False, metavar='TF',
                    help='save individual images')

args = parser.parse_args()

for (path, dirs, files) in os.walk(args.pathFrom):
    print(path)
    targetDir = os.path.join(args.pathTo, path[len(args.pathFrom) + 1:])
    if not os.path.exists(targetDir):
        os.mkdir(targetDir)
    if len(dirs) == 0:
        pack = {}
        n = 0
        for fileName in files:
            (idx, ext) = os.path.splitext(fileName)
            if ext == '.png':
                png = sio.imread(os.path.join(path, fileName))
                tensor = torch.Tensor(png.astype(float)).byte()
                if args.split:
                    torch.save(tensor, os.path.join(targetDir, idx + '.pt'))
                else:
                    pack[int(idx.split('x')[0])] = tensor
                n += 1
                if n % 100 == 0:
                    print('Converted ' + str(n) + ' images.')
        if len(pack) > 0:
            torch.save(pack, targetDir + '/pack.pt')
            print('Saved pt binary.')
            del pack
