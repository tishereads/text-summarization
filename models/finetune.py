"""Fine-tune a seq2seq summarization model (PEGASUS / T5 / BART) on CNN/DailyMail or XSum."""
import argparse
import os

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data.load_datasets import get_columns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="google/pegasus-large",
                   help="google/pegasus-large, google/pegasus-xsum, t5-small, facebook/bart-base, ...")
    p.add_argument("--dataset", type=str, choices=["cnn_dailymail", "xsum"], default="cnn_dailymail")
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_target_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--num_train_samples", type=int, default=None)   # e.g. 20000 for a quick run
    p.add_argument("--num_eval_samples", type=int, default=1000)
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    input_col, target_col = get_columns(args.dataset)
    model_name_tag = args.model_name.replace("/", "-")

    # ---------- Load tokenizer & model ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.config.max_length = args.max_target_length

    # ---------- Load & preprocess data ----------
    train_ds = load_dataset(args.dataset, split="train")
    eval_ds = load_dataset(args.dataset, split="validation")
    if args.num_train_samples:
        train_ds = train_ds.select(range(min(args.num_train_samples, len(train_ds))))
    eval_ds = eval_ds.select(range(min(args.num_eval_samples, len(eval_ds))))

    def preprocess(examples):
        inputs = [doc for doc in examples[input_col]]
        model_inputs = tokenizer(
            inputs, max_length=args.max_input_length, truncation=True, padding=False
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[target_col], max_length=args.max_target_length, truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

    # ---------- ROUGE metric ----------
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects newline-separated sentences
        decoded_preds = ["\n".join(p.strip().split(". ")) for p in decoded_preds]
        decoded_labels = ["\n".join(l.strip().split(". ")) for l in decoded_labels]
        result = rouge.compute(
            predictions=decoded_preds, references=decoded_labels,
            use_stemmer=True,
        )
        return {k: round(v, 4) for k, v in result.items()}

    # ---------- Trainer ----------
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, model_name_tag),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rouge2",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, model_name_tag, "final"))
    print("Training complete. Model saved to", os.path.join(args.output_dir, model_name_tag, "final"))


if __name__ == "__main__":
    main()
