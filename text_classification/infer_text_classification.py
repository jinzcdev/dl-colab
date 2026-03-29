#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""加载 finetune 的 output 目录，用 ModelScope 做文本分类推理。"""

import argparse
import json
import os
import sys

from modelscope.pipelines import pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="文本分类推理")
    ap.add_argument("--model_dir", required=True, help="work_dir/output，含 configuration.json")
    ap.add_argument("--text", help="单条文本（与 --input_file 二选一）")
    ap.add_argument("--input_file", help="批量：每行一句，或 JSONL（--text_field 取字段）")
    ap.add_argument("--text_field", default="sentence", help="JSONL 中的文本字段名")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--output_file", help="可选，JSONL 写出路径")
    ap.add_argument("--first_sequence", help="不填则从 configuration.json 读 preprocessor")
    args = ap.parse_args()

    if bool(args.text) == bool(args.input_file):
        ap.error("--text 与 --input_file 必须且只能选一个")

    model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    cfg = os.path.join(model_dir, "configuration.json")
    if not os.path.isfile(cfg):
        sys.exit(f"缺少 {cfg}")
    # 与训练时 preprocessor 的 first_sequence 对齐
    with open(cfg, encoding="utf-8") as f:
        prep = json.load(f).get("preprocessor") or {}
    first_seq = args.first_sequence or prep.get("first_sequence") or "text"

    clf = pipeline(
        "text-classification", model=model_dir, device=args.device, first_sequence=first_seq
    )

    fout = open(args.output_file, "w", encoding="utf-8") if args.output_file else None
    try:

        def dump(obj):  # 打印并可选写入 JSONL
            s = json.dumps(obj, ensure_ascii=False)
            print(s)
            if fout:
                fout.write(s + "\n")

        if args.text:
            r = clf(args.text, topk=args.topk)  # 单条推理
            dump({"text": args.text, "labels": r["labels"], "scores": r["scores"]})
        else:
            # 组 batch 后一次 forward，比逐条调用更快
            texts, nrows = [], []
            with open(args.input_file, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                        t = o[args.text_field] if isinstance(o, dict) and args.text_field in o else line
                    except json.JSONDecodeError:
                        t = line
                    texts.append(t)
                    nrows.append((i, t))
            if not texts:
                sys.exit("输入文件无有效行")
            for (no, t), r in zip(
                nrows, clf(texts, batch_size=args.batch_size, topk=args.topk)
            ):
                dump({"line_no": no, "text": t, "labels": r["labels"], "scores": r["scores"]})
    finally:
        if fout:
            fout.close()


if __name__ == "__main__":
    main()
