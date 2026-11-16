"""Eval: exact-match odgovora na chat JSONL (held-out ili proxy uzorak). Bez QC-a ne pokretati na held-outu izvan evala.
Koristenje: python training/eval.py --adapter releases/adapter-v01 --base <bazni-model> --eval data/synth/val.jsonl
"""
import argparse, json
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="opcionalno; bez njega se evaluira cisti base (baseline)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = svi")
    a = ap.parse_args()
    proc = AutoProcessor.from_pretrained(a.base)
    model = AutoModelForImageTextToText.from_pretrained(a.base, device_map="auto")
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    ok, n = 0, 0
    for line in open(a.eval):
        line = line.strip()
        if not line:
            continue
        if a.max_samples and n >= a.max_samples:
            break
        ex = json.loads(line)
        msgs, gold = ex["messages"][:-1], ex["messages"][-1]["content"][0]["text"]
        imgs = [Image.open(p).convert("RGB") for p in ex["images"]]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=text, images=imgs, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=64)
        pred = proc.decode(out[0][inp["input_ids"].shape[1]:]).strip()
        n += 1
        ok += (pred == gold.strip())
        print(f"[{n}] gold={gold.strip()!r} pred={pred!r}")
    print(f"exact-match: {ok}/{n} = {ok/max(n,1):.3f}")


if __name__ == "__main__":
    main()
