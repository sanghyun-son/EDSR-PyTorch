# EDSR baseline model (x2) + JPEG augmentation
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --data_train DIV2K+DIV2K-Q75 --data_test DIV2K+DIV2K-Q75

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save edsr_baseline_x4 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]









# EDSR in the paper (x4) - from EDSR (x2)
# FULL DATA
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../models/EDSR_x4.pt --save_models  --data_train DIV2K --dir_data ../../dataset --data_range 1-1002/1-1 --chop --epochs 25 --loss 1*L1C

# TRAIN/TEST
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../models/EDSR_x4.pt --save_models  --data_train DIV2K --dir_data ../../dataset --data_range 1-900/901-1002 --chop --epochs 25 --loss 1*L1C --lr 5e-4
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/edsr_x4/model/model_lr5e4/model_10.pt --save_models  --data_train DIV2K --dir_data ../../dataset --data_range 1-900/901-1002 --chop --epochs 25 --loss 1*L1C --lr 7e-5


#python main.py --data_test DIV2K --data_range 1-801/802-901 --dir_data ../../dataset --model EDSR --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/edsr_x4/model/model_best.pt --test_only --chop
#python main.py --data_test DIV2K --data_range 1-801/802-901 --dir_data ../../dataset --model EDSR --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/edsr_x4/model/model_1.pt --test_only --chop


# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --data_test Demo --scale 4 --save rcan_x4 --patch_size 192 --pre_train ../models/RCAN_BIX2.pt --save_models  --data_train DIV2K --dir_data ../../dataset --data_range 1-1002/1-1 --epochs 15 --loss 1*L1 --chop



# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble





# Test your own images
#python main.py --data_test Demo --model EDSR --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/edsr_x4/model/model_20epoch_lr7e5_from_10epoch_lr5e4/model_20.pt --test_only --save_results --chop
#python main.py --data_test Demo --model EDSR --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/edsr_x4/model/model_20.pt --test_only --save_results --chop

#python main.py --data_test Demo --model EDSR --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train  ../models/EDSR_x4_l1_5ep.pt --test_only --save_results --chop

#python main.py --data_test Demo --model MDSR --scale 4 --pre_train download --test_only --save_results

# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
python main.py --template RCAN --data_test Demo --scale 4 --test_only --save_results --patch_size 192 --pre_train ../models/rcan_2_7_13_13.pt
#python main.py --template RCAN --data_test DIV2K --dir_data ../../dataset --data_range 1-900/901-1000 --scale 4 --test_only --patch_size 192 --pre_train ../models/rcan_2.pt



# Advanced - Test with JPEG images 
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download

# RDN BI model (x2)
#python3.6 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 64 --reset
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt

