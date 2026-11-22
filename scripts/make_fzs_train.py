"""FZS trening set: federalna CPI saopcenja 01-08/2026 (drugi layout od bhas-a).
Svaki QA se provjerava: odgovor mora doslovno postojati u tekstualnom sloju stranice.
Koristenje: SRC=/tmp/opencode/fzs IMG=data/fzs-train/img OUT=data/fzs-train python3 scripts/make_fzs_train.py
"""
import fitz
import json
import os
import re

SRC = os.getenv("SRC", "/tmp/opencode/fzs")
IMG = os.getenv("IMG", "data/fzs-train/img")
OUT = os.getenv("OUT", "data/fzs-train")
os.makedirs(IMG, exist_ok=True)

NOM = ["januar", "februar", "mart", "april", "maj", "juni", "juli", "avgust",
       "septembar", "oktobar", "novembar", "decembar"]
LOC = {"januar": "januaru", "februar": "februaru", "mart": "martu", "april": "aprilu",
       "maj": "maju", "juni": "junu", "juli": "julu", "avgust": "avgustu",
       "septembar": "septembru", "oktobar": "oktobru", "novembar": "novembru",
       "decembar": "decembru"}
RIM = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

rows = []


def must2(text, ans, ctx):
    if ans not in text and ans.replace(",", ".") not in text and ans.replace(".", ",") not in text:
        raise ValueError(f"odgovor nije u tekstu [{ctx}]: {ans!r}")


def qa(img, q, a, text, ctx):
    must2(text, a, ctx)
    rows.append({"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
        {"role": "assistant", "content": [{"type": "text", "text": a}]}],
        "images": [img]})


def numbers_after(text, label, n=7):
    i = text.lower().find(label.lower())
    if i < 0:
        return []
    seg = text[i + len(label):i + len(label) + 150]
    return re.findall(r"\d{1,4}[.,]\d{1,2}%?", seg)[:n]


def main():
    import glob
    for pdf in sorted(glob.glob(f"{SRC}/FZS_2026_*.pdf")):
        mm = os.path.basename(pdf)[9:11]  # FZS_2026_08.pdf -> 08
        mi = int(mm) - 1
        mj, loc, rim = NOM[mi], LOC[NOM[mi]], RIM[mi]
        tag = f"fz_{mm}"
        d = fitz.open(pdf)
        pages = [" ".join(p.get_text().split()) for p in d]
        t1 = pages[0]
        d[0].get_pixmap(dpi=150).save(f"{IMG}/{tag}_p1.png")
        img1 = f"data/fzs-train/img/{tag}_p1.png"
        ctx0 = f"fzs-2026-{mm}"
        # naslovnica: datum, broj
        m = re.search(r"SARAJEVO,\s*(\d{2}\.\d{2}\.\d{4})\.", t1)
        if m:
            qa(img1, "Kojeg datuma je objavljeno saopćenje? Odgovori samo datumom.", m.group(1), t1, ctx0)
        m = re.search(r"BROJ / NUMBER:\s*(7\.1\.\d+)", t1)
        if m:
            qa(img1, "Koji je broj saopćenja? Odgovori samo oznakom.", m.group(1), t1, ctx0)
        # narativ: mjesecni i godisnji rast
        m = re.search(r"u prosjeku su porasle za ([\d,]+%) u odnosu na prethodni mjesec", t1)
        if m:
            qa(img1, f"Za koliko su u prosjeku porasle cijene u {loc} 2026. u odnosu na prethodni mjesec? Odgovori samo procentom.",
               m.group(1), t1, ctx0)
        m = re.search(r"u odnosu na \w+/?\w* 2025[^.]*porasle za ([\d,]+%) \(godišnja", t1)
        if m:
            qa(img1, f"Za koliko su porasle cijene u {loc} 2026. u odnosu na isti mjesec 2025.? Odgovori samo procentom.",
               m.group(1), t1, ctx0)
        m = re.search(r"Prijevoz za ([\d,]+%)", t1)
        if m:
            qa(img1, "Za koliko su mjesečno porasle cijene prijevoza? Odgovori samo procentom.", m.group(1), t1, ctx0)
        # tabela (str 2-3): isti COICOP kostur
        t = pages[1] + " " + (pages[2] if len(pages) > 2 else "")
        d[1].get_pixmap(dpi=150).save(f"{IMG}/{tag}_p2.png")
        if len(pages) > 2:
            d[2].get_pixmap(dpi=150).save(f"{IMG}/{tag}_p3.png")
        for label, q in [("Ukupan Indeks", "ukupni indeks"),
                         ("Hrana i bezalkoholna pića", "indeks hrane i bezalkoholnih pića"),
                         ("Prijevoz", "indeks prijevoza"),
                         ("Odjeća i obuća", "indeks odjeće i obuće"),
                         ("Zdravstvo", "indeks zdravstva")]:
            v = numbers_after(t, label)
            if len(v) >= 2:
                qa(f"data/fzs-train/img/{tag}_p2.png",
                   f"Koliki je {q} za {rim} 2026. u odnosu na prethodni mjesec? Odgovori samo brojem.",
                   v[0], t, ctx0 + "-t1")
                qa(f"data/fzs-train/img/{tag}_p2.png",
                   f"Koliki je {q} za {rim} 2026. u odnosu na isti mjesec 2025.? Odgovori samo brojem.",
                   v[1], t, ctx0 + "-t2")
    with open(f"{OUT}/train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"gotovo: uzoraka={len(rows)}")


if __name__ == "__main__":
    main()
