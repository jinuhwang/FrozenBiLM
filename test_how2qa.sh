##!/bin/bash

# # TRANSFORMERS_CACHE=/workspace/ssd/cache \
# TRANSFORMERS_CACHE=microsoft \
# python \
#     -m torch.distributed.launch --nproc_per_node 4 --use_env \
#     mc.py \
#     --test --eval \
#     --combine_datasets how2qa \
#     --combine_datasets_val how2qa \
#     --how2qa_val_csv_path /mnt/hdd2/how2qa/frozenbilm/public_val_sanitized.csv \
#     --how2qa_subtitles_path /mnt/hdd2/how2qa/frozenbilm/subtitles.pkl \
#     --save_dir=zsHow2QA \
#     --ds_factor_ff=8 \
#     --ds_factor_attn=8 \
#     --suffix="." \
#     --batch_size_val=32 \
#     --max_tokens=512 \
#     --load=/workspace/data/frozenbilm/checkpoints/frozenbilm_how2qa.pth

# for NOISE in 0.90 0.92 0.94 0.96 0.98
# for NOISE in 0.50 0.60 0.70 0.75 0.80 0.85 0.90 0.92 0.94 0.95 0.96 0.97 0.98 0.99
# 0.01 step from 0.50 to 1.
for NOISE in $(seq 0.50 0.01 1.00)
do
    echo "Noise: $NOISE"
    for i in $(seq 1 30)
    do
        INJECT_NOISE=$NOISE \
        TRANSFORMERS_CACHE=microsoft \
        python \
            -m torch.distributed.launch --nproc_per_node 4 --use_env \
            mc.py \
            --test --eval \
            --combine_datasets how2qa \
            --combine_datasets_val how2qa \
            --how2qa_val_csv_path /mnt/hdd2/how2qa/frozenbilm/public_val_sanitized.csv \
            --how2qa_subtitles_path /mnt/hdd2/how2qa/frozenbilm/subtitles.pkl \
            --save_dir=zsHow2QA \
            --ds_factor_ff=8 \
            --ds_factor_attn=8 \
            --suffix="." \
            --batch_size_val=32 \
            --max_tokens=512 \
            --load=/workspace/data/frozenbilm/checkpoints/frozenbilm_how2qa.pth
    done
done
