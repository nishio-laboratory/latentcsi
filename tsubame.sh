#!/bin/sh

# カレントディレクトリでジョブを実行する場合に指定
#$ -cwd

# 資源タイプを指定（必須）
#$ -l gpu_1=1
# 実行時間を指定（必須；お試し実行機能は利用最長時間10分）
#$ -l h_rt=23:59:59
# ジョブ名を指定（任意）
#$ -N tsubame_training_seg

module load cuda
module load intel

nvidia-smi
python3 -m pip install --user torch torchvision numpy diffusers transformers accelerate lightning more-itertools

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 src/encoder/training_seg.py -p /gs/fs/tga-nlab/eshan/csi_image_data/datasets/walking -b 16 -e 200 -s


wait
