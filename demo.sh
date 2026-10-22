# To download and prepare data 
python3 prepare_LR_data.py

# To Generate bicubic data 
# python3 prepare_bicubic_data.py

# To do EDSR training for scale 2 
# python train_edsr_x2_ucmerced.py --data-dir data/edsr_data --seed 42 --batch-size 4 --epochs 100 --patience 15
#
# To do EDSR training for scale 3 
# python train_edsr_x3_ucmerced.py --data-dir data/edsr_data --seed 42 --batch-size 4 --epochs 100 --patience 15
#
# To do EDSR training for scale 4 
# python train_edsr_x4_ucmerced.py --data-dir data/edsr_data --seed 42 --batch-size 4 --epochs 100 --patience 15
#
# To do LW-EDSR training for scale 2 
# python train_light_weight_ucmerced.py --scale 2 --data-dir data/edsr_data --seed 42 --batch-size 4 --epochs 100 --patience 15
#
# To do LW-EDSR training for scale 3 
# python train_light_weight_ucmerced.py --scale 3 --data-dir data/edsr_data --seed 42 --batch-size 4 --epochs 100 --patience 15
#
# To do LW-EDSR training for scale 4 
# python train_light_weight_ucmerced.py --scale 4 --data-dir data/edsr_data --seed 42 --batch-size 4 --epochs 100 --patience 15
#
# Train resnet for semantic loss 
# python3 train_resnet18_x1_ucmerced.py


### Resnet 18 must be trained and weights need to be produced before training for semantic weights
# To do LW-EDSR-SL training for scale 2 
# python train_light_weight_ucmerced.py --scale 2 --seed 42 --data-dir data/edsr_data --epochs 100 --batch-size 4 --semantic-loss --semantic-weights weights/resnet18_ucmerced_x1_best.pth
#
# To do LW-EDSR-SL training for scale 3 
# python train_light_weight_ucmerced.py --scale 3 --seed 42 --data-dir data/edsr_data --epochs 100 --batch-size 4 --semantic-loss --semantic-weights weights/resnet18_ucmerced_x1_best.pth
#
# To do LW-EDSR-SL training for scale 4 
# python train_light_weight_ucmerced.py --scale 4 --seed 42 --data-dir data/edsr_data --epochs 100 --batch-size 4 --semantic-loss --semantic-weights weights/resnet18_ucmerced_x1_best.pth
#
# To classify all the images using trained resnet model 
# python3 evaluate_sr_classification.py
