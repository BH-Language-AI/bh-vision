"""QC gate: blokira trening ako manifest sadrži nečiste licence (oba sloja: anotacije + slike)."""
import sys

BLOCKED = {"gpl-3.0", "cc-by-nc", "cc-by-nc-sa", "non-commercial", "rail", "research-only"}

def main(manifest="data/manifest.yaml"):
    try:
        text = open(manifest).read().lower()
    except FileNotFoundError:
        print(f"QC: nema {manifest} — trening BLOKIRAN dok se ne doda manifest sa licencama.")
        sys.exit(2)
    bad = [b for b in BLOCKED if b in text]
    if bad:
        print(f"QC BLOKIRAN: nedozvoljene licence {bad} (ChartQA/FUNSD/XFUND/RVL-CDIP/IAM/LAION/Salamandra-VL).")
        sys.exit(1)
    print("QC OK: licence čiste (očekivano: cc-by-4.0/cc0/cdla-permissive + provjerene slike).")

if __name__ == "__main__":
    main(sys.argv[2] if len(sys.argv) > 2 else "data/manifest.yaml")
