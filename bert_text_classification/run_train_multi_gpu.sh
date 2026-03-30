#!/usr/bin/env bash
# 多卡训练：使用 torchrun 启动多进程（每进程一张 GPU），依赖 train.py 中 WORLD_SIZE>1 时传入 launcher
set -euo pipefail

# 进程数（通常等于使用的 GPU 数量）；示例：NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_train_multi_gpu.sh
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

PYTHONPATH=. torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" bert_text_classification/train.py
