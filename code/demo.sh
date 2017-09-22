# Demo code for training

# Training EDSR_baseline_x2
#python main.py --template EDSR --model EDSR --scale 2 --nFeat 64 --nResBlock 16 --patchSize 96 --load EDSR_baseline_x2 --reset True

# Training EDSR_baseline_x3
#python main.py --template EDSR --model EDSR --scale 3 --nFeat 64 --nResBlock 16 --patchSize 144 --load EDSR_baseline_x3

# Training EDSR_baseline_x4
#python main.py --template EDSR --model EDSR --scale 4 --nFeat 64 --nResBlock 16 --patchSize 192 --load EDSR_baseline_x4

# Training MDSR_baseline
#python main.py --template MDSR --model MDSR --scale 2+3+4 --patchSize 48 --nFeat 64 --nResBlock 16 --load MDSR_baseline --reset True

# Training EDSR_x2
#python main.py --template EDSR --model EDSR_scale --scale 2 --nFeat 256 --nResBlock 32 --patchSize 96 --load EDSR_x2 --reset True

# Training EDSR_x3
#python main.py --template EDSR --model EDSR_scale --scale 3 --nFeat 256 --nResBlock 32 --patchSize 144 --load EDSR_x3 --reset True

# Training EDSR_x4
#python main.py --template EDSR --model EDSR_scale --scale 4 --nFeat 256 --nResBlock 32 --patchSize 192 --load EDSR_x4 --reset True

# Training MDSR
#python main.py --template MDSR --model MDSR --scale 2+3+4 --patchSize 48 --nFeat 64 --nResBlock 80 --load MDSR --reset True



# Demo code for test (Examples)

# Test with MDSR_baseline
#python main.py  --testData myImage --scale 2+3+4 --preTrained ../experiment/model/MDSR_baseline.pt --testOnly True --saveResults True --save test_MDSR_baseline --reset True

# Test with MDSR_baseline_jpeg
#python main.py  --testData myImage --scale 2+3+4 --preTrained ../experiment/model/MDSR_baseline_jpeg.pt --testOnly True --saveResults True --save test_MDSR_baseline_jpeg --reset True

# Test with EDSR_x4
#python main.py  --testData myImage --scale 4 --preTrained ../experiment/model/EDSR_baseline_x4.pt --testOnly True --saveResults True --save test_EDSR_x4 --reset True
