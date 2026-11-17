"""BH-Vision MINI: SmolVLM2-500M QLoRA na v03, 24 koraka + eval 10 val uzoraka (adapter vs ocekivano).
Stane u ostatak kvote (~1h). Verzije pinovane (tf 5.22.2)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import subprocess

def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh("pip install -q transformers==5.22.2 trl==1.18.0 peft==0.21.2 accelerate datasets pillow huggingface_hub num2words bitsandbytes 2>&1 | tail -1")

import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)
assert torch.cuda.is_available(), "nema GPU-a, prekidam"

from huggingface_hub import snapshot_download
BASE = snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
DATA = snapshot_download("BH-Language-AI/bs-vision-train-v03", repo_type="dataset", allow_patterns=["data/synth3/*"])

from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import torch.nn.functional as F
from PIL import Image
import json

def ce_loss(outputs, labels, num_items_in_batch):
    logits = outputs.logits.float()
    sl = logits[..., :-1, :].contiguous()
    lb = labels[..., 1:].contiguous()
    return F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1), ignore_index=-100)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForImageTextToText.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
proc = AutoProcessor.from_pretrained(BASE)
if hasattr(model, "vision_model"):
    for p in model.vision_model.parameters():
        p.requires_grad = False

ds = load_dataset("json", data_files=f"{DATA}/data/synth3/train.jsonl", split="train")
ds = ds.map(lambda ex: {"images": [f"{DATA}/{p}" for p in ex["images"]]})
print("train rows:", len(ds), flush=True)

def fmt(ex):
    return {"text": proc.apply_chat_template(ex["messages"], tokenize=False), "images": ex["images"]}
dstr = ds.map(fmt, remove_columns=[c for c in ds.column_names if c != "images"])

args = SFTConfig(output_dir="/kaggle/working/output", max_steps=24,
                 learning_rate=2e-4, per_device_train_batch_size=1,
                 gradient_accumulation_steps=8, logging_steps=4, save_steps=24,
                 max_length=None, dataset_text_field="text", remove_unused_columns=False)
SFTTrainer(model=model, train_dataset=dstr, args=args,
           peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
           processing_class=proc, compute_loss_func=ce_loss).train()
model.save_pretrained("/kaggle/working/output/mini-500m")
print("MINI TRENING GOTOV", flush=True)

# eval 10 val uzoraka istim adapterom
model.eval()
ok, n = 0, 0
for line in open(f"{DATA}/data/synth3/val.jsonl"):
    line = line.strip()
    if not line or n >= 10:
        break
    ex = json.loads(line)
    gold = ex["messages"][-1]["content"][0]["text"]
    imgs = [Image.open(p).convert("RGB") for p in [f"{DATA}/{p}" for p in ex["images"]]]
    text = proc.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
    inp = proc(text=text, images=imgs, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=32)
    pred = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    norm = lambda s: s.strip().rstrip(".").casefold()
    hit = norm(pred) == norm(gold.strip())
    ok += hit
    n += 1
    print(f"[{n}] gold={gold.strip()!r} pred={pred.strip()!r} {'OK' if hit else '--'}", flush=True)
print(f"MINI EVAL: {ok}/{n} = {ok/max(n,1):.3f} (baseline netreniranog 2.2B bio 7/10)", flush=True)
