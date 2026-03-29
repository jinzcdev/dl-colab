#!/usr/bin/env python
"""
在 DAMO_NLP/yf_dianping 数据集上微调文本分类（StructBERT / BERT 系 backbone）。

要点（与 ModelScope 官方 examples/pytorch/text_classification 一致）：
1. model 传 ModelScope Hub 上的预训练 backbone，权重与 tokenizer 从快照目录加载；
2. cfg_file 指向本目录 configuration.json，只描述 task / train / preprocessor 等训练侧配置；
3. cfg_modify_fn 从训练集+验证集统计 label2id，并同步 model.num_labels 与 LinearLR 步数。

运行：
  bash examples/pytorch/bert_text_classification/run_train_single_gpu.sh
  bash examples/pytorch/bert_text_classification/run_train_multi_gpu.sh
"""

import os

from modelscope import EpochBasedTrainer
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile

# 预训练中文句向量/分类常用 backbone（StructBERT，与官方 text_classification 示例一致）
# 若需其它 BERT 系模型，可改为 Hub 上带 text-classification 或 backbone 的 model id
PRETRAINED_MODEL_ID = 'damo/nlp_structbert_backbone_base_std'

# work_dir 必须放在模型目录之外：CheckpointHook 会 copy 模型目录内容，避免递归嵌套
WORK_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'tmp', 'structbert_text_classification'))
# 本目录的 configuration.json：训练超参、preprocessor 类型等（不含 pytorch 权重）
CFG_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), ModelFile.CONFIGURATION)

# 与 MsDataset 中样本字段一致；yf_dianping 常见为评论文本 + 标签，若列名为 text 请改为 "text"
FIRST_SEQUENCE_KEY = 'sentence'
LABEL_KEY = 'label'


def set_labels(labels):
    """根据标签列表构建 label -> id 映射（与 finetune_text_classification 逻辑一致）。"""
    if isinstance(labels, str):
        label_list = labels.split(',')
    else:
        unique_labels = set(labels)
        label_list = list(unique_labels)
        label_list.sort(key=lambda x: str(x))
        label_list = list(
            map(lambda x: x if isinstance(x, str) else str(x), label_list))
    return {label: idx for idx, label in enumerate(label_list)}


def build_cfg_modify_fn(train_ds: MsDataset, eval_ds: MsDataset):
    """返回 cfg_modify_fn：写入 label2id、num_labels、LinearLR total_iters。"""

    def cfg_modify_fn(cfg):
        # 从整列读取标签并合并训练/验证，保证 label2id 覆盖两边出现的类别
        labels = list(train_ds[LABEL_KEY]) + list(eval_ds[LABEL_KEY])
        label2id = set_labels(labels)
        cfg.merge_from_dict({'preprocessor.label2id': label2id})
        cfg.preprocessor.first_sequence = FIRST_SEQUENCE_KEY
        cfg.preprocessor.label = LABEL_KEY
        cfg.model.num_labels = len(label2id)

        if cfg.evaluation.period.eval_strategy == 'by_epoch':
            cfg.evaluation.period.by_epoch = True

        if cfg.train.lr_scheduler.type == 'LinearLR':
            bs = cfg.train.dataloader.batch_size_per_gpu
            steps_per_epoch = max(1, int(len(train_ds) / bs))
            cfg.train.lr_scheduler.total_iters = steps_per_epoch * cfg.train.max_epochs

        return cfg

    return cfg_modify_fn


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    train_dataset = MsDataset.load('DAMO_NLP/yf_dianping', split='train', subset_name='default')
    eval_dataset = MsDataset.load('DAMO_NLP/yf_dianping', split='validation', subset_name='default')

    cfg_modify_fn = build_cfg_modify_fn(train_dataset, eval_dataset)

    kwargs = dict(
        model=PRETRAINED_MODEL_ID,
        cfg_file=CFG_FILE,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        seed=42,
        work_dir=WORK_DIR,
        cfg_modify_fn=cfg_modify_fn,
    )
    # torchrun 会设置 WORLD_SIZE>1，需传入 launcher 以注册 DDPHook（与 finetune_text_classification 一致）
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        kwargs['launcher'] = 'pytorch'
    # NLP 任务建议使用 nlp_base_trainer：Tokenizer 构建等与官方单测一致
    trainer: EpochBasedTrainer = build_trainer(
        name=Trainers.nlp_base_trainer,
        default_args=kwargs,
    )
    trainer.train()
    print('训练结束，日志与 checkpoint 目录:', WORK_DIR)


if __name__ == '__main__':
    main()
