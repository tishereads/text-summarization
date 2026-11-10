"""Evaluate a saved model on a test split and report ROUGE-1/2/L."""
import argparse

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data.load_datasets import get_columns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dataset", type=str, choices=["cnn_dailymail", "xsum"], default="cnn_dailymail")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_target_length", type=int, default=128)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    input_col, target_col = get_columns(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.select(range(min(args.num_samples, len(ds))))

    rouge = evaluate.load("rouge")
    all_preds, all_refs = [], []

    for i in tqdm(range(0, len(ds), args.batch_size)):
        batch = ds[i : i + args.batch_size]
        inputs = tokenizer(
            batch[input_col], max_length=args.max_input_length,
            truncation=True, padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            summaries = model.generate(
                **inputs,
                max_length=args.max_target_length,
                num_beams=4,
                length_penalty=0.8,   # PEGASUS-style setting
            )
        preds = tokenizer.batch_decode(summaries, skip_special_tokens=True)
        all_preds.extend(["\n".join(p.strip().split(". ")) for p in preds])
        all_refs.extend(["\n".join(r.strip().split(". ")) for r in batch[target_col]])

    result = rouge.compute(predictions=all_preds, references=all_refs, use_stemmer=True)
    print("\n=== ROUGE on", args.dataset, "===")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
