"""Preuzmi plotqa-bs skup s Hugging Facea (prevelik za git).
Koristenje: python3 scripts/fetch_plotqa_bs.py
Cilj: data/plotqa-bs/{train.jsonl,img/*.jpg} — iste relativne putanje kao u jsonl-u.
"""
from huggingface_hub import snapshot_download

DST = "data/plotqa-bs"
snapshot_download("BH-Language-AI/bs-vision-plotqa-bs-v01", repo_type="dataset",
                  local_dir=DST,
                  allow_patterns=["train.jsonl", "img/*.jpg", "README.md"])
print("skinuto u", DST)
