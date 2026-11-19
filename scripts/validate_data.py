"""Validira image-text-to-text chat JSONL po training/DATA_CONTRACT.md.
Koristenje: python3 scripts/validate_data.py --data <jsonl> [--images] (provjera da slike postoje i PIL ih otvara)
"""
import json, os, sys


def main(path, check_images=False):
    from PIL import Image
    bad, n = 0, 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            r = json.loads(line)
            msgs = r["messages"]
            assert isinstance(msgs, list) and msgs, "prazne messages"
            assert "images" in r, "nedostaje images"
            if check_images:
                for p in r["images"]:
                    assert os.path.exists(p), f"nema slike: {p}"
                    Image.open(p).load()
        except Exception as e:  # noqa
            bad += 1
            print(f"red {n}: {e}")
    print(f"validate: {n-bad}/{n} OK ({path})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[args.index("--data") + 1] if "--data" in args else "data/train.jsonl"
    main(path, check_images="--images" in args)
