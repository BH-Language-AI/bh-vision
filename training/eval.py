"""Eval: exact-match odgovora na chat JSONL (held-out ili proxy uzorak). Bez QC-a ne pokretati na held-outu izvan evala.
Koristenje: python training/eval.py --adapter releases/adapter-v01 --base <bazni-model> --eval data/synth/val.jsonl
"""
import argparse, json, re
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from PIL import Image


def norm_strict(s):
    return s.strip().rstrip(".").casefold()


def norm_nums(s):
    return [x.replace(",", ".") for x in re.findall(r"-?\d+[.,]?\d*", s.replace("%", ""))]


def hit_lenient(pred, gold):
    """Blaza metrika: isti brojevi (zarez==tacka) ILI gold podstring predikcije. Stroga ostaje primarna."""
    g, p = gold.strip().casefold(), pred.strip().casefold()
    if norm_strict(pred) == norm_strict(gold.strip()):
        return True
    if norm_nums(g) and norm_nums(g) == norm_nums(p):
        return True
    if len(g) > 3 and g in p:
        return True
    return g in ("da", "ne") and g == p.rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="opcionalno; bez njega se evaluira cisti base (baseline)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = svi")
    ap.add_argument("--out", default=None, help="opcionalno: snimi rezultate u JSON")
    ap.add_argument("--system", default="", help="opcionalna sistemska instrukcija (prefiks svim pitanjima)")
    ap.add_argument("--trust-remote-code", action="store_true", help="samo za modele bez nativne podrske (Pixtral)")
    a = ap.parse_args()
    proc = AutoProcessor.from_pretrained(a.base, trust_remote_code=a.trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(a.base, device_map="auto", trust_remote_code=a.trust_remote_code)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    ok, n, loose = 0, 0, 0
    for line in open(a.eval):
        line = line.strip()
        if not line:
            continue
        if a.max_samples and n >= a.max_samples:
            break
        ex = json.loads(line)
        msgs, gold = ex["messages"][:-1], ex["messages"][-1]["content"][0]["text"]
        if a.system:
            msgs = [{"role": "system", "content": [{"type": "text", "text": a.system}]}] + msgs
        imgs = [Image.open(p).convert("RGB") for p in ex["images"]]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=text, images=imgs, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=64)
        pred = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        n += 1
        hit = norm_strict(pred) == norm_strict(gold.strip())
        ok += hit
        loose += hit_lenient(pred, gold)
        print(f"[{n}] gold={gold.strip()!r} pred={pred.strip()!r} {'OK' if hit else '--'}", flush=True)
    print(f"exact-match (normalized): {ok}/{n} = {ok/max(n,1):.3f}")
    print(f"lenient (brojevi/zarez/podstring): {loose}/{n} = {loose/max(n,1):.3f}")
    if a.out:
        import json as _json
        with open(a.out, "w") as f:
            _json.dump({"model": a.adapter or a.base, "eval": a.eval,
                        "ok": ok, "n": n, "acc": ok/max(n,1),
                        "loose_ok": loose, "loose_acc": loose/max(n,1)}, f, indent=1)
        print("snimljeno u", a.out)


if __name__ == "__main__":
    main()
