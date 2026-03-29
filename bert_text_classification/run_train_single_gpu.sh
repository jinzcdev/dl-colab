#!/usr/bin/env bash
# 单卡训练：在 ModelScope 仓库根目录下执行 text_classification.py
set -euo pipefail

# 默认使用环境可见 GPU；只跑一张卡时可设：CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python bert_text_classification/text_classification.py
