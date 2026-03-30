# 大纲

- 平台介绍
  - huggingface.co
  - modelscope.cn (魔搭平台)
  - 为什么选择 modelscope
- 示例介绍
  - 使用 modelscope 平台训练 BERT 模型实现文本分类
  - 原因：BERT 是 Transformer 架构在 NLP 领域应用的经典模型，而且大家对自然语言处理相关的模型接触较多一些
  - 数据集：DAMO_NLP/yf_dianping (大众点评评论数据集)
- 环境准备
  - 个人服务器
    - 操作系统：Linux (Ubuntu较多)
    - 显卡（必须）：8GB 显存以上
    - Python 环境
      - conda
      - 虚拟环境
  - modelscope 平台集成服务器
    - 直接使用魔搭平台的免费开发环境（自带 CUDA 环境）
