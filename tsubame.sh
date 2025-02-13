#!/bin/sh

# カレントディレクトリでジョブを実行する場合に指定
#$ -cwd

# 資源タイプを指定（必須）
#$ -l cpu_80=1
# 実行時間を指定（必須；お試し実行機能は利用最長時間10分）
#$ -l h_rt=23:59:59
# ジョブ名を指定（任意）
#$ -N opensvdd-ton-ids

module load cuda
module load intel

nvidia-smi
python3 -m pip install --user uv
uv pip install torch torchvision numpy diffusers transformers accelerate lightning more-itertools

python src/encoder/training_seg.py -p /gs/fs/tga-nlab/eshan/datasets/walking
wait
