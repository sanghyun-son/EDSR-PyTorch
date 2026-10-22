for scale in 4 3 2; do
    for seed in 999 123 42; do
        echo "========================================"
        echo "Training lightweight EDSR: x${scale}, seed ${seed}"
        echo "========================================"

        python train_light_weight_ucmerced.py \
            --scale "$scale" \
            --seed "$seed" \
            --data-dir data/edsr_data \
            --epochs 100 \
            --batch-size 4 \
        || { echo "FAILED: x${scale}, seed ${seed}"; exit 1; }
    done
done
