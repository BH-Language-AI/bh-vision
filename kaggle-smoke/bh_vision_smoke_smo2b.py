"""BH-Vision SMOKE: SmolVLM2-2.2B QLoRA, 32 uzorka, 20 koraka. Samo verifikacija pipelinea."""
import json, os, subprocess, sys

def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh("pip install -q transformers==5.22.2 trl==1.18.0 peft==0.21.2 accelerate datasets pillow huggingface_hub num2words bitsandbytes 2>&1 | tail -1")

import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)
assert torch.cuda.is_available(), "nema GPU-a, prekidam"

from huggingface_hub import snapshot_download
BASE = snapshot_download("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
DATA = snapshot_download("BH-Language-AI/bs-vision-train-v02", repo_type="dataset", allow_patterns=["data/synth2/*"])
print("base:", BASE, "\ndata:", DATA, flush=True)

from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

ds = load_dataset("json", data_files=f"{DATA}/data/synth2/train.jsonl", split="train").select(range(32))
# preusmjeri relativne putanje slika na skinuti snapshot
def fix(ex):
    ex["images"] = [f"{DATA}/{p}" for p in ex["images"]]
    return ex
ds = ds.map(fix)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForImageTextToText.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
proc = AutoProcessor.from_pretrained(BASE)
if hasattr(model, "vision_model"):
    for p in model.vision_model.parameters():
        p.requires_grad = False

def fmt(ex):
    return {"text": proc.apply_chat_template(ex["messages"], tokenize=False), "images": ex["images"]}
ds = ds.map(fmt, remove_columns=[c for c in ds.column_names if c != "images"])

args = SFTConfig(output_dir="/kaggle/working/output", max_steps=20,
                 learning_rate=2e-4, per_device_train_batch_size=1,
                 gradient_accumulation_steps=8, logging_steps=5, save_steps=20,
                 max_length=None, dataset_text_field="text", remove_unused_columns=False)
trainer = SFTTrainer(model=model, train_dataset=ds, args=args,
                     peft_config=LoraConfig(r=8, lora_alpha=16, target_modules="all-linear",
                                            task_type="CAUSAL_LM"),
                     processing_class=proc)
res = trainer.train()
print("SMOKE OK — final loss:", res.training_loss, flush=True)
model.save_pretrained("/kaggle/working/output/adapter-smoke")
print("adapter sacuvan", flush=True)
