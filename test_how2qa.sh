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

# Testing synthetic cosine similarity noise
# for NOISE in $(seq 0.505 0.005 0.895)
# do
#     echo "Noise: $NOISE"
#     for i in $(seq 1 10)
#     do
#         INJECT_NOISE=$NOISE \
#         TRANSFORMERS_CACHE=microsoft \
#         python \
#             -m torch.distributed.launch --nproc_per_node 4 --use_env \
#             mc.py \
#             --test --eval \
#             --combine_datasets how2qa \
#             --combine_datasets_val how2qa \
#             --how2qa_val_csv_path /mnt/hdd2/how2qa/frozenbilm/public_val_sanitized.csv \
#             --how2qa_subtitles_path /mnt/hdd2/how2qa/frozenbilm/subtitles.pkl \
#             --save_dir=zsHow2QA \
#             --ds_factor_ff=8 \
#             --ds_factor_attn=8 \
#             --suffix="." \
#             --batch_size_val=32 \
#             --max_tokens=512 \
#             --load=/workspace/data/frozenbilm/checkpoints/frozenbilm_how2qa.pth
#     done
# done

# Testing synthetics MSE noise
for NOISE in $(seq 0.05 0.05 0.5)
do
    echo "Noise: $NOISE"
    for i in $(seq 1 5)
    do
	INJECT_NOISE_MSE=1 \
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
