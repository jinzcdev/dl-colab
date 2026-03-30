import os

from modelscope.pipelines import pipeline

MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'tmp', 'structbert_text_classification/output'))


def main():

    remark_list = [
        "这家店的饭菜还不错的，值得再来",
        "这家店的饭菜一般的，不来也罢",
        "真是无语了，这家店",
        "位置偏僻，交通不便，但是还挺好吃的",
        "价格有点贵，但是量很足，味道不错",
        "服务态度很好，但是上菜速度有点慢",
        "环境不错，但是有点吵闹",
        "位置偏僻，服务很差，环境也一般，除了味道不错其他的都不行",
        "位置很近，服务很好，环境很好，价格也便宜，但是就菜不好吃"
    ]

    # 大众点评分类模型
    dianping_clf = pipeline(
        "text-classification", model=MODEL_DIR, device="cuda", first_sequence="sentence",
    )

    results = dianping_clf(remark_list, topk=1, batch_size=16)
    for remark, result in zip(remark_list, results):
        print(
            f"remark: {remark}, score: {result['scores']}, label: {'好' if result['labels'][0] == '1' else '差'}")


if __name__ == "__main__":
    main()
