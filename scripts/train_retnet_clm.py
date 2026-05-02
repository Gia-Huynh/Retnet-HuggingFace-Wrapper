#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import math
import os
from functools import partial

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from retnet_hf import RetNetConfig, RetNetForCausalLM

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TorchScale RetNet wrapped as a Hugging Face CausalLM.")
    parser.add_argument("--dataset_name", type=str, default="FiscalNote/billsum")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_text_field", type=str, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="outputs/retnet-billsum")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=1024)
    parser.add_argument("--validation_split_percentage", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_retention_heads", type=int, default=8)
    parser.add_argument("--value_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation_dropout", type=float, default=0.0)
    parser.add_argument("--drop_path_rate", type=float, default=0.0)
    parser.add_argument("--recurrent_chunk_size", type=int, default=128)
    parser.add_argument("--chunkwise_recurrent", action="store_true")
    return parser.parse_args()


def infer_text_field(dataset_name: str, dataset_text_field: str | None) -> str | None:
    if dataset_text_field:
        return dataset_text_field
    lower = dataset_name.lower()
    if "billsum" in lower:
        return None  # handled by formatter below
    if "c4" in lower:
        return "text"
    return "text"


def load_raw_datasets(args: argparse.Namespace):
    lower = args.dataset_name.lower()
    if "billsum" in lower:
        ds = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" in ds:
            train_ds, eval_ds = ds["train"], ds["validation"]
        else:
            if args.streaming:
                raise ValueError("Streaming Billsum validation split creation is not supported in this example.")
            split = ds["train"].train_test_split(
                test_size=args.validation_split_percentage / 100.0,
                seed=args.seed,
            )
            train_ds, eval_ds = split["train"], split["test"]
        return train_ds, eval_ds

    if "c4" in lower:
        config_name = args.dataset_config_name or "en"
        train_ds = load_dataset(args.dataset_name, config_name, split="train", streaming=args.streaming)
        eval_ds = load_dataset(args.dataset_name, config_name, split="validation", streaming=args.streaming)
        return train_ds, eval_ds

    ds = load_dataset(args.dataset_name, args.dataset_config_name)
    if "validation" in ds:
        return ds["train"], ds["validation"]
    if args.streaming:
        raise ValueError("Streaming is only wired here for datasets that already expose validation splits.")
    split = ds["train"].train_test_split(
        test_size=args.validation_split_percentage / 100.0,
        seed=args.seed,
    )
    return split["train"], split["test"]


def format_for_causal_lm(example, dataset_name: str, text_field: str | None):
    lower = dataset_name.lower()
    if "billsum" in lower:
        return {
            "text": f"Document:\n{example['text']}\n\nSummary:\n{example['summary']}"
        }
    if text_field is None:
        raise ValueError("Could not infer dataset text field. Pass --dataset_text_field.")
    return {"text": example[text_field]}


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples, block_size: int):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = list(result["input_ids"])
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    set_seed(args.seed)

    text_field = infer_text_field(args.dataset_name, args.dataset_text_field)
    train_raw, eval_raw = load_raw_datasets(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    format_fn = partial(format_for_causal_lm, dataset_name=args.dataset_name, text_field=text_field)

    remove_columns = getattr(train_raw, "column_names", None)
    train_ds = train_raw.map(format_fn, remove_columns=remove_columns)
    eval_ds = eval_raw.map(format_fn, remove_columns=getattr(eval_raw, "column_names", None))

    tokenize_fn = partial(tokenize_function, tokenizer=tokenizer)
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    if args.max_train_samples is not None and not args.streaming:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples is not None and not args.streaming:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    group_fn = partial(group_texts, block_size=args.block_size)
    train_ds = train_ds.map(group_fn, batched=True)
    eval_ds = eval_ds.map(group_fn, batched=True)

    config = RetNetConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_retention_heads=args.num_retention_heads,
        value_dim=args.value_dim,
        hidden_dropout_prob=args.dropout,
        activation_dropout=args.activation_dropout,
        drop_path_rate=args.drop_path_rate,
        recurrent_chunk_size=args.recurrent_chunk_size,
        chunkwise_recurrent=args.chunkwise_recurrent,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = RetNetForCausalLM(config)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        fp16=args.fp16,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()

    print("Saved model to", os.path.abspath(args.output_dir))
    print("Eval metrics:", metrics)


if __name__ == "__main__":
    main()
