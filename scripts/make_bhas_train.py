"""bhas train set: avgust/juli 2026 saopcenja (isti layouti kao held-out septembar, DRUGI brojevi).
Svaki QA se provjerava: odgovor mora doslovno postojati u tekstualnom sloju stranice.
Koristenje: SRC=/tmp/opencode/heldout-train IMG=data/bhas-train/img OUT=data/bhas-train python3 scripts/make_bhas_train.py
"""
import fitz
import json
import os
import re

SRC = os.getenv("SRC", "/tmp/opencode/heldout-train")
IMG = os.getenv("IMG", "data/bhas-train/img")
OUT = os.getenv("OUT", "data/bhas-train")
os.makedirs(IMG, exist_ok=True)

DOCS = {
    # tag: (pdf, mjesec_ime, mjesec_broj, godina, datum, broj)
    "01_08": ("PRI_01_2026_08_1_BS.pdf", "avgust", "VIII", "2026", "25.09.2026.", "08"),
    "01_07": ("PRI_01_2026_07_1_BS.pdf", "juli", "VII", "2026", "25.08.2026.", "07"),
    "02_08": ("PRI_02_2026_08_1_BS.pdf", "avgust", "VIII", "2026", "25.09.2026.", "08"),
    "04_08": ("PRI_04_2026_08_1_BS.pdf", "avgust", "VIII", "2026", "25.09.2026.", "08"),
    "04_07": ("PRI_04_2026_07_1_BS.pdf", "juli", "VII", "2026", "26.08.2026.", "07"),
}
PREV = {"VIII": "VII", "VII": "VI"}

NUM = r"\d{1,4}(?:[.,]\d{1,2})?%?"
COMMA_DOCS = ("b_01", "b_04")  # ovi dokumenti na slici koriste zarez-decimale
UNITS = {"kg", "l", "komad", "kutija", "par", "m3", "m2", "kwh", "t", "komplet",
         "porcija", "tretman", "pakiranje", "paket", "bočica", "set", "minuta",
         "uplata", "pregled", "karta", "stranica", "mjesec", "mjesecna", "godišnja"}


def parse_prices(pdf_path, pages):
    """Izvuci (label, unit, price) trojke iz cjenovnih tabela: unit-red, broj-red, label je red iznad."""
    import fitz as _fz
    DROP = {"mjesečna", "godišnja", "uplata", "feniranjem", "stranica", "uplata,"}
    out = []
    d = _fz.open(pdf_path)
    for pg in pages:
        lines = [l.strip() for l in d[pg - 1].get_text().split("\n")]
        for i, ln in enumerate(lines):
            if ln in UNITS and i + 1 < len(lines) and re.fullmatch(r"\d+[.,]\d+", lines[i + 1]) and i > 0:
                lab = lines[i - 1]
                if lab and not re.fullmatch(r"[\d.,/% ]+", lab) and lab not in UNITS and len(lab) > 2 \
                        and lab.lower() not in DROP:
                    out.append((lab, ln, lines[i + 1], pg))
    return out


def norm(tag, ans):
    """Uskladi decimalni znak s onim na slici (01/04 zarez, 02 tacka)."""
    if tag.startswith(COMMA_DOCS) and re.fullmatch(r"\d+\.\d+%?", ans):
        return ans.replace(".", ",")
    return ans


def numbers_after(text, label, n=6):
    """Nadji label u tekstu, vrati sljedece brojeve."""
    i = text.find(label)
    if i < 0:
        return []
    seg = text[i + len(label):i + len(label) + 120]
    return re.findall(r"\d{1,4}[.,]\d{1,2}%?|\d{2,4}[.,]\d", seg)[:n]


def must2(text, ans, ctx):
    if ans not in text and ans.replace(",", ".") not in text and ans.replace(".", ",") not in text:
        raise ValueError(f"odgovor nije u tekstu [{ctx}]: {ans!r}")
    return ans


rows = []
qi = [0]


def qa(img, q, a, text, ctx):
    tag = img.split("/")[-1][:4]  # b_01 / b_02 / b_04
    a = norm(tag, a)
    must2(text, a.replace(",", "."), ctx)
    rows.append({"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
        {"role": "assistant", "content": [{"type": "text", "text": a}]}],
        "images": [img]})
    qi[0] += 1


def render(tag, pdf, page1, dst):
    d = fitz.open(f"{SRC}/{pdf}")
    p = d[page1 - 1]
    p.get_pixmap(dpi=170).save(dst)
    return ' '.join(p.get_text().split())


def title_q(tag, img, text, title, m, ctx):
    qa(img, "Koji je naslov dokumenta? Odgovori doslovno.", title, text, ctx)
    d = DOCS[tag][4]
    qa(img, "Kojeg datuma je objavljeno saopćenje? Odgovori samo datumom.", d, text, ctx)
    b = DOCS[tag][5]
    qa(img, "Koji je broj saopćenja? Odgovori samo brojem.", b, text, ctx)


def main():
    # ---- 01_08 CPI avgust ----
    t = render("01_08", *DOCS["01_08"][:1], 1, f"{IMG}/b_01_08_p1.png")
    title_q("01_08", f"data/bhas-train/img/b_01_08_p1.png", t,
            "Indeks potrošačkih cijena u Bosni i Hercegovini, avgust 2026", "avgust", "01_08")
    m = re.search(r"u prosjeku su više za ([\d,]+%)", t)
    qa(f"data/bhas-train/img/b_01_08_p1.png",
       "Za koliko su u prosjeku više cijene u avgustu 2026. u odnosu na prethodni mjesec? Odgovori samo procentom.",
       m.group(1), t, "01_08-mj")
    m = re.search(r"isti mjesec prethodne godine viši je za ([\d,]+%)", t)
    qa(f"data/bhas-train/img/b_01_08_p1.png",
       "Za koliko su više cijene u avgustu 2026. u odnosu na isti mjesec prethodne godine? Odgovori samo procentom.",
       m.group(1), t, "01_08-god")
    m = re.search(r"Prijevoz za ([\d,]+%)", t)
    qa(f"data/bhas-train/img/b_01_08_p1.png",
       "Za koliko su mjesečno porasle cijene prijevoza? Odgovori samo procentom.", m.group(1), t, "01_08-pr")
    # ---- 01_07 CPI juli ----
    t = render("01_07", *DOCS["01_07"][:1], 1, f"{IMG}/b_01_07_p1.png")
    title_q("01_07", f"data/bhas-train/img/b_01_07_p1.png", t,
            "Indeks potrošačkih cijena u Bosni i Hercegovini, juli 2026", "juli", "01_07")
    m = re.search(r"u prosjeku su niže za ([\d,]+%)", t)
    qa(f"data/bhas-train/img/b_01_07_p1.png",
       "Za koliko su u prosjeku niže cijene u julu 2026. u odnosu na prethodni mjesec? Odgovori samo procentom.",
       m.group(1), t, "01_07-mj")
    # ---- 01_08 p2 CPI tabela ----
    t = render("01_08", *DOCS["01_08"][:1], 2, f"{IMG}/b_01_08_p2.png")
    for label, q in [("Ukupan indeks", "ukupni indeks"),
                     ("Hrana i bezalkoholna pića", "hrane i bezalkoholnih pića"),
                     ("Prijevoz", "prijevoza"),
                     ("Odjeća i obuća", "odjeće i obuće"),
                     ("Zdravstvo", "zdravstva")]:
        v = numbers_after(t, label)
        if len(v) >= 2:
            qa(f"data/bhas-train/img/b_01_08_p2.png",
               f"Koliki je {q} za VIII 2026. u odnosu na VII 2026.? Odgovori samo brojem.",
               v[0], t, f"01_08-tab-{label}")
            qa(f"data/bhas-train/img/b_01_08_p2.png",
               f"Koliki je {q} za VIII 2026. u odnosu na VIII 2025.? Odgovori samo brojem.",
               v[1], t, f"01_08-tab2-{label}")
    # ---- 01_07 p2 CPI tabela ----
    t = render("01_07", *DOCS["01_07"][:1], 2, f"{IMG}/b_01_07_p2.png")
    for label, q in [("Ukupan indeks", "ukupni indeks"),
                     ("Hrana i bezalkoholna pića", "hrane i bezalkoholnih pića"),
                     ("Prijevoz", "prijevoza"),
                     ("Odjeća i obuća", "odjeće i obuće")]:
        v = numbers_after(t, label)
        if len(v) >= 2:
            qa(f"data/bhas-train/img/b_01_07_p2.png",
               f"Koliki je {q} za VII 2026. u odnosu na VI 2026.? Odgovori samo brojem.",
               v[0], t, f"01_07-tab-{label}")
    # ---- 02_08 naslovnica ----
    t = render("02_08", *DOCS["02_08"][:1], 1, f"{IMG}/b_02_08_p1.png")
    title_q("02_08", "data/bhas-train/img/b_02_08_p1.png", t,
            "Prosječne potrošačke cijene u Bosni i Hercegovini u avgustu 2026. godine", "avgust", "02_08")
    # ---- 02_08 cijene: SVI proizvodi s obje stranice (parser) ----
    import fitz as _fz2
    _d = _fz2.open(f"{SRC}/PRI_02_2026_08_1_BS.pdf")
    for label, unit, price, pg in parse_prices(f"{SRC}/PRI_02_2026_08_1_BS.pdf", [1, 2]):
        t = " ".join(_d[pg - 1].get_text().split())
    for label, unit, price, pg in parse_prices(f"{SRC}/PRI_02_2026_08_1_BS.pdf", [1, 2]):
        t = fitz.open(f"{SRC}/PRI_02_2026_08_1_BS.pdf")[pg - 1].get_text()
        t = " ".join(t.split())
        qa(f"data/bhas-train/img/b_02_08_p{pg}.png",
           f"Kolika je prosječna cijena ({label.lower()})? Odgovori samo brojem.",
           price, t, f"02_08-sve-{label[:20]}")
    # ---- 04_08 PPI narativ p1 ----
    t = render("04_08", *DOCS["04_08"][:1], 1, f"{IMG}/b_04_08_p1.png")
    title_q("04_08", f"data/bhas-train/img/b_04_08_p1.png", t,
            "Indeks proizvođačkih cijena industrije u Bosni i Hercegovini u avgustu 2026. godine", "avgust", "04_08")
    for pat, q in [(r"ukupnu industriju su zabilježile rast nivoa za ([\d,]+%), dok",
                    "Za koliko su mjesečno porasle proizvođačke cijene za ukupnu industriju? Odgovori samo procentom."),
                   (r"isti mjesec prethodne godine zabilježile rast za ([\d,]+%)",
                    "Za koliko su godišnje porasle proizvođačke cijene za ukupnu industriju? Odgovori samo procentom."),
                   (r"domaćem tržištu su.*?rast nivoa za ([\d,]+%)",
                    "Za koliko su mjesečno porasle cijene na domaćem tržištu? Odgovori samo procentom."),
                   (r"nedomaćem tržištu su.*?rast nivoa za ([\d,]+%)",
                    "Za koliko su mjesečno porasle cijene na nedomaćem tržištu? Odgovori samo procentom.")]:
        m = re.search(pat, t)
        if m:
            qa(f"data/bhas-train/img/b_04_08_p1.png", q, m.group(1), t, f"04_08-{q[:20]}")
    # ---- 04_08 p4 PPI tabela ----
    t = render("04_08", *DOCS["04_08"][:1], 4, f"{IMG}/b_04_08_p4.png")
    for label, q, idx in [("UKUPAN INDEKS", "ukupan indeks na domaćem tržištu za VIII 2026. u odnosu na VII 2026.", 0),
                          ("UKUPAN INDEKS", "ukupan indeks na ukupnom tržištu za VIII 2026. u odnosu na VIII 2025.", 5),
                          ("Energija", "indeks energije na nedomaćem tržištu za VIII 2026. u odnosu na VIII 2025.", 3),
                          ("Kapitalni proizvodi", "indeks kapitalnih proizvoda na domaćem tržištu za VIII 2026. u odnosu na VII 2026.", 0)]:
        v = numbers_after(t, label)
        if len(v) > idx:
            qa(f"data/bhas-train/img/b_04_08_p4.png", f"Koliki je {q}? Odgovori samo brojem.", v[idx], t, f"04_08-tab-{label}-{idx}")
    # ---- 04_07 PPI narativ p1 ----
    t = render("04_07", *DOCS["04_07"][:1], 1, f"{IMG}/b_04_07_p1.png")
    title_q("04_07", f"data/bhas-train/img/b_04_07_p1.png", t,
            "Indeks proizvođačkih cijena industrije u Bosni i Hercegovini u julu 2026. godine", "juli", "04_07")
    for pat, q in [(r"ukupnu industriju su zabilježile rast nivoa za ([\d,]+%), dok",
                    "Za koliko su mjesečno porasle proizvođačke cijene za ukupnu industriju? Odgovori samo procentom."),
                   (r"isti mjesec prethodne godine zabilježile rast za ([\d,]+%)",
                    "Za koliko su godišnje porasle proizvođačke cijene za ukupnu industriju? Odgovori samo procentom.")]:
        m = re.search(pat, t)
        if m:
            qa(f"data/bhas-train/img/b_04_07_p1.png", q, m.group(1), t, f"04_07-{q[:20]}")

    with open(f"{OUT}/train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"gotovo: uzoraka={len(rows)}")


if __name__ == "__main__":
    main()
