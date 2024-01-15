#!/bin/bash

    # Original accuracy
    # --how2qa_features_path /mnt/ssd2/dataset/how2qa/openai_clip-vit-large-patch14 \

    # No visual features
    # --how2qa_features_path /dev/null \

    # DiffRate Only
    # --how2qa_features_path /mnt/ssd2/dataset/how2qa/diffrate/openai_clip-vit-large-patch14/p_257_256_252_244_241_237_233_230_225_222_220_200_155_105_81_65_60_56_52_50_49_48_33_5_m_257_253_247_241_240_235_231_228_223_222_216_181_122_87_65_62_56_54_50_49_48_47_25_5 \

    # Reues Only
    # --how2qa_features_path /mnt/ssd2/dataset/how2qa/reuse/openai_clip-vit-large-patch14/try23 \

if [[ $# -ne 1 ]] ; then
    echo 'Usage: test_how2qa.sh how2qa/tryXXX'
    exit 1
fi

MODEL=$1
TRANSFORMERS_CACHE=microsoft \
python \
    mc.py \
    --test --eval \
    --combine_datasets how2qa \
    --combine_datasets_val how2qa \
    --how2qa_val_csv_path /mnt/ssd2/dataset/how2qa/how2qa_frozenbilm_sanitized.csv \
    --how2qa_subtitles_path /mnt/hdd2/how2qa/frozenbilm/subtitles.pkl \
    --how2qa_features_path /mnt/ssd2/dataset/how2qa/reuse/openai_clip-vit-large-patch14/${MODEL} \
    --save_dir=zsHow2QA \
    --ds_factor_ff=8 \
    --ds_factor_attn=8 \
    --suffix="." \
    --batch_size_val=32 \
    --max_tokens=512 \
    --load=/workspace/data/frozenbilm/checkpoints/frozenbilm_how2qa.pth

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

# # Testing synthetics MSE noise
# for NOISE in $(seq 0.00 0.005 0.15)
# do
#     echo "Noise: $NOISE"
#     for i in $(seq 1 5)
#     do
# 	INJECT_NOISE_MSE=1 \
#         INJECT_NOISE=$NOISE \
#         TRANSFORMERS_CACHE=microsoft \
#         python \
#             -m torch.distributed.launch --nproc_per_node 4 --use_env \
#             mc.py \
#             --test --eval \
#             --combine_datasets how2qa \
#             --combine_datasets_val how2qa \
#             --how2qa_val_csv_path /mnt/hdd2/how2qa/frozenbilm/public_val_sanitized.csv \
#             --how2qa_features_path /mnt/ssd2/dataset/how2qa/openai_clip-vit-large-patch14 \
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
