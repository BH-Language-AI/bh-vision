import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

BASE = "/home/sha/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
AD = "/home/sha/.cache/huggingface/hub/models--BH-Language-AI--bh-vision-1-4b-v2/snapshots/0a11ae995fd1ff68fe90f2b68cf69db014d73b4e"
OUT = "/tmp/opencode/bh-vision-1-4b-v2-merged"

print("ucitavam bazu...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    BASE, dtype=torch.float32, device_map="cpu", trust_remote_code=True)
print("ucitavam adapter...", flush=True)
model = PeftModel.from_pretrained(model, AD)
print("spajam...", flush=True)
model = model.merge_and_unload()
model.save_pretrained(OUT)
print("spremljeno", flush=True)
proc = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
proc.save_pretrained(OUT)
print("MERGE-OK", flush=True)
