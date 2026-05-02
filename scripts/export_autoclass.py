#!/usr/bin/env python
from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from retnet_hf import RetNetConfig, RetNetForCausalLM, RetNetModel


def main():
    parser = argparse.ArgumentParser(description="Register the custom RetNet wrapper for Hugging Face AutoClasses.")
    parser.add_argument("model_dir", type=str)
    args = parser.parse_args()

    RetNetConfig.register_for_auto_class()
    RetNetModel.register_for_auto_class("AutoModel")
    RetNetForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = RetNetForCausalLM.from_pretrained(args.model_dir)
    model.save_pretrained(args.model_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
    except Exception:
        pass

    # smoke-test local load through AutoModelForCausalLM
    AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True)


if __name__ == "__main__":
    main()
