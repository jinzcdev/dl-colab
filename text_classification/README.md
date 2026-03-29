# 文本分类（ModelScope）

本目录包含基于 ModelScope 的**微调训练**与**本地 checkpoint 推理**脚本，与仓库根目录 `requirements.txt` 中的依赖一致。

## 训练

训练脚本将 checkpoint 与配置写入 `work_dir/output/`，其中至少包含：

- `configuration.json`：任务与 preprocessor 配置（含 `label2id`）
- `pytorch_model.bin`：权重
- `config.json`、`vocab.txt` 等 backbone 相关文件

示例（与 `run_train.sh` 一致）：

```bash
cd /path/to/bert-demo
PYTHONPATH=. python text_classification/finetune_text_classification.py \
    --task 'text-classification' \
    --model 'damo/nlp_structbert_backbone_base_std' \
    --train_dataset_name 'clue' \
    --val_dataset_name 'clue' \
    --train_subset_name 'tnews' \
    --val_subset_name 'tnews' \
    --train_split 'train' \
    --val_split 'validation' \
    --first_sequence 'sentence' \
    --label label \
    --preprocessor 'sen-cls-tokenizer' \
    --use_model_config True \
    --work_dir './tmp/my_experiment' \
    --max_epochs 12 \
    --per_device_train_batch_size 32 \
    --eval_metrics 'seq-cls-metric'
```

训练结束后，推理应指向 **`work_dir/output`**（不是 `work_dir` 根目录）。

## 推理

使用 `infer_text_classification.py` 加载上述 `output` 目录，调用 ModelScope `text-classification` pipeline。

### 依赖

```bash
pip install -r requirements.txt
```

### 单条文本

```bash
cd /path/to/bert-demo
PYTHONPATH=. python text_classification/infer_text_classification.py \
    --model_dir ./tmp/my_experiment/output \
    --text "这是一条待分类的新闻标题或正文" \
    --device cpu \
    --topk 5
```

标准输出为一行 JSON，包含 `text`、`labels`（类别名或 id）、`scores`（对应 softmax 概率，由高到低）。

### 批量文件

- **纯文本**：每行一条样本。
- **JSONL**：每行一个 JSON 对象，通过 `--text_field` 指定文本字段（CLUE `tnews` 与训练脚本一致时常为 `sentence`）。

```bash
PYTHONPATH=. python text_classification/infer_text_classification.py \
    --model_dir ./tmp/my_experiment/output \
    --input_file ./samples.jsonl \
    --text_field sentence \
    --batch_size 16 \
    --device gpu \
    --output_file ./predictions.jsonl
```

### 常用参数说明

| 参数 | 说明 |
|------|------|
| `--model_dir` | 微调产物目录，必须为含 `configuration.json` 的 `output` 路径 |
| `--device` | `gpu` / `cpu` / `cuda:0` 等 |
| `--topk` | 返回前 k 个类别及概率 |
| `--first_sequence` | 若不指定，则从 `configuration.json` 的 `preprocessor.first_sequence` 读取，与训练一致 |
| `--batch_size` | 仅批量模式有效 |

若更换数据集或自定义 JSON 字段名，请同时调整训练时的 `--first_sequence` 与推理时的 `--text_field` / `--first_sequence`，保证与 preprocessor 期望的输入键一致。

## 相关文件

| 文件 | 作用 |
|------|------|
| `finetune_text_classification.py` | 微调入口 |
| `infer_text_classification.py` | 本地 checkpoint 推理 |
| `run_train.sh` | 单机训练示例命令 |
| `run_train_dist.sh` | 分布式训练示例命令 |
