# TorchScale RetNet -> Hugging Face wrapper

This project wraps the uploaded TorchScale-style `RetNetDecoder` into a Hugging Face custom model so you can:

- instantiate it with a `RetNetConfig`
- train it with `Trainer` as a causal LM
- save it with `save_pretrained()`
- optionally register AutoClasses and push/load it with `trust_remote_code=True`

The wrapper is built directly around the uploaded `retnet.py`, which exposes `RetNetDecoder` and its recurrent decoding path. The file defines the decoder layers, recurrence handling through `incremental_state`, and the relative-position logic used by the model. fileciteturn4file0

## What is included

- `retnet_hf/configuration_retnet.py` — Hugging Face config class
- `retnet_hf/modeling_retnet.py` — Hugging Face `RetNetModel` and `RetNetForCausalLM`
- `retnet_hf/retnet.py` — your uploaded TorchScale RetNet file copied into the package
- `scripts/train_retnet_clm.py` — training script; defaults to BillSum
- `scripts/export_autoclass.py` — rewrites `config.json` with AutoClass metadata for Hub sharing

## Install

TorchScale documents installation via `pip install torchscale`; the repository README also shows that path. Hugging Face custom model support is based on `PreTrainedConfig` and `PreTrainedModel`, which is the pattern used here. citeturn291520search1turn525709view0

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train on BillSum

BillSum is available on the Hugging Face Hub as `FiscalNote/billsum`, a text summarization dataset in the 10K-100K size range. The script converts each example into a single autoregressive training string:

```text
Document:
<bill text>

Summary:
<reference summary>
```

That makes it usable with a causal language-modeling objective. The Hugging Face causal LM guide trains next-token models with the left-to-right objective used here, and Hugging Face datasets can be loaded with `load_dataset()`. citeturn649846view0turn649846view1turn806500search4

```bash
python scripts/train_retnet_clm.py \
  --dataset_name FiscalNote/billsum \
  --tokenizer_name gpt2 \
  --output_dir outputs/retnet-billsum \
  --block_size 512 \
  --hidden_size 512 \
  --intermediate_size 2048 \
  --num_hidden_layers 8 \
  --num_retention_heads 8 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-4 \
  --num_train_epochs 1 \
  --eval_steps 200 \
  --save_steps 200
```

## Train on C4

C4 is much larger. Hugging Face notes that loading all of C4 can take a long time and that the dataset is extremely large; their docs recommend loading a subset via `data_files` or using streaming for very large corpora. This is why the training script includes a `--streaming` flag. citeturn649846view1turn649846view2

```bash
python scripts/train_retnet_clm.py \
  --dataset_name allenai/c4 \
  --dataset_config_name en \
  --streaming \
  --tokenizer_name gpt2 \
  --output_dir outputs/retnet-c4 \
  --block_size 512 \
  --max_steps 2000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16
```

## Export as a Hub-ready custom model

Hugging Face documents custom-model support through `PreTrainedConfig` / `PreTrainedModel`, `register_for_auto_class()`, and loading with `trust_remote_code=True`. The helper below updates the saved model directory to include the AutoClass metadata. citeturn525709view0turn525709view1

```bash
python scripts/export_autoclass.py outputs/retnet-billsum
```

Then locally:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "outputs/retnet-billsum",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("outputs/retnet-billsum")
```

## Notes and limitations

1. This is a wrapper around the uploaded decoder file, not a full port of every training utility in TorchScale.
2. The exact TorchScale dependency versions matter. If a newer `torchscale` release changes constructor args or component signatures, update the config-to-args mapping in `modeling_retnet.py`.
3. The environment I built this in did not have `transformers`, `datasets`, `torchscale`, or `fairscale` installed, so I could not execute an end-to-end training run here. The code is structured to match the documented Hugging Face custom-model and causal-LM patterns, but you should still do a local smoke test after installing dependencies. citeturn525709view0turn649846view0
4. The wrapper exposes recurrent caching through `past_key_values` by forwarding TorchScale's `incremental_state` dictionary, so generation uses the decoder's recurrent mode defined in the uploaded file. fileciteturn4file0
