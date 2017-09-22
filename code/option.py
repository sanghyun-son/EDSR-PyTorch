import argparse
import template

parser = argparse.ArgumentParser(description='Semantic aware super-resolution')

parser.add_argument('--template', default='.', metavar='TMP',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--nThreads', type=int, default=3, metavar='N',
                    help='number of threads for data loading')
parser.add_argument('--cuda', default=True, metavar='TF',
                    help='enables CUDA training')
parser.add_argument('--nGPUs', type=int, default=1, metavar='N',
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')

# Data specifications
parser.add_argument('--task', default='SR', metavar='TASK',
                    help='which task to perform')
parser.add_argument('--dataDir', default='../../../dataset', metavar='DIR',
                    help='dataset directory')
parser.add_argument('--trainData', default='DIV2K', metavar='NAME',
                    help='train dataset name')
parser.add_argument('--testData', default='DIV2K', metavar='NAME',
                    help='test dataset name')
parser.add_argument('--nTrain', type=int, default=800, metavar='N',
                    help='number of training set')
parser.add_argument('--nVal', type=int, default=10, metavar='N',
                    help='number of validation set')
parser.add_argument('--valOffset', type=int, default=800, metavar='N',
                    help='validation index offest')
parser.add_argument('--ext', default='ptpack', metavar='EXT',
                    help='dataset file extension')
parser.add_argument('--scale', default='4', metavar='S',
                    help='super resolution scale')
parser.add_argument('--patchSize', type=int, default=192, metavar='N',
                    help='output patch size')
parser.add_argument('--rgbRange', type=int, default=255, metavar='R',
                    help='maximum value of RGB')
parser.add_argument('--nChannel', type=int, default=3, metavar='C',
                    help='number of color channels to use')
parser.add_argument('--quality', default='', metavar='Q',
                    help='jpeg compression quality')
parser.add_argument('--hFlip', default=True, metavar='TF',
                    help='data augmentation (horizontal flip)')
parser.add_argument('--rot', default=True, metavar='TF',
                    help='data augmentation (rotation)')

# Model specifications
parser.add_argument('--model', default='EDSR', metavar='MODEL',
                    help='model name')
parser.add_argument('--preTrained', default='.', metavar='PRE',
                    help='pre-trained model directory')
parser.add_argument('--extendFrom', default='.', metavar='EXTEND',
                    help='pre-trained model directory')
parser.add_argument('--nResBlock', type=int, default=16, metavar='B',
                    help='number of residual blocks')
parser.add_argument('--nFeat', type=int, default=64, metavar='B',
                    help='number of feature maps')
parser.add_argument('--subMean', default=False, metavar='TF',
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', default='single', metavar='FP',
                    help='model and data precision')
parser.add_argument('--multiOutput', default=False, metavar='FP',
                    help='model generates multiple outputs')
parser.add_argument('--multiTarget', default=False, metavar='FP',
                    help='model requires multiple targets')

# Training specifications
parser.add_argument('--reset', default=False, metavar='TF',
                    help='reset the training')
parser.add_argument('--testEvery', type=int, default=1000, metavar='N',
                    help='do test per every N batches')
parser.add_argument('--testOnly', default=False, metavar='TF',
                    help='set this option to test the model')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batchSize', type=int, default=16, metavar='N',
                    help='input batch size for training')
parser.add_argument('--selfEnsemble', default=False, metavar='TF',
                    help='use self-ensemble method for test')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--lrDecay', type=int, default=40, metavar='N',
                    help='learning rate decay per N epochs')
parser.add_argument('--decayType', default='step', metavar='TYPE',
                    help='learning rate decay type (step | exp | inv)')
parser.add_argument('--decayFactor', type=int, default=2, metavar='N',
                    help='learning rate decay factor for step decay')

parser.add_argument('--optimizer', default='ADAM', metavar='O',
                    help='optimizer to use')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9, metavar='B',
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, metavar='B',
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8, metavar='eps',
                    help='ADAM epsilon for numerical stability')

# Loss specifications
parser.add_argument('--loss', default='1*L1', metavar='L',
                    help='loss function configuration')

# Log specifications
parser.add_argument('--save', default='test', metavar='NAME',
                    help='file name to save')
parser.add_argument('--load', default='.', metavar='NAME',
                    help='file name to load')
parser.add_argument('--printModel', default=False, metavar='TF',
                    help='print model')
parser.add_argument('--saveModel', type=int, default=-1, metavar='N',
                    help='extract trained model from selected epoch')
parser.add_argument('--printEvery', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--saveResults', default=False, metavar='TF',
                    help='save output results')

args = parser.parse_args()
template.setTemplate(args)

args.task = args.task.split('+')
args.testData = args.testData.split('+')

args.scale = args.scale.split('+')
for i, s in enumerate(args.scale):
    args.scale[i] = int(s)

args.quality = args.quality.split('+')
for i, q in enumerate(args.quality):
    if q != '':
        args.quality[i] = int(q)

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
