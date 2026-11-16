"""Validira image-text-to-text chat JSONL po training/DATA_CONTRACT.md."""
import json, sys

def main(path):
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
        except Exception as e:  # noqa
            bad += 1
            print(f"red {n}: {e}")
    print(f"validate: {n-bad}/{n} OK ({path})")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main(sys.argv[2] if len(sys.argv) > 2 else "data/train.jsonl")
