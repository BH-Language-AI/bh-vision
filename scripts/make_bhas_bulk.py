"""bhas bulk: svi mjeseci (osim held-out 09/2026) -> trening QA.
Metapodaci (mjesec/godina/broj/datum/naslov) se vade s naslovnice; pitanja ista kao heldout.
Koristenje: SRC=/tmp/opencode/bhas-bulk IMG=data/bhas-bulk/img OUT=data/bhas-bulk python3 scripts/make_bhas_bulk.py
"""
import fitz
import json
import os
import re

SRC = os.getenv("SRC", "/tmp/opencode/bhas-bulk")
IMG = os.getenv("IMG", "data/bhas-bulk/img")
OUT = os.getenv("OUT", "data/bhas-bulk")
os.makedirs(IMG, exist_ok=True)

NOM = ["januar", "februar", "mart", "april", "maj", "juni", "juli",
       "avgust", "septembar", "oktobar", "novembar", "decembar"]
LOC = {"januar": "januaru", "februar": "februaru", "mart": "martu", "april": "aprilu",
       "maj": "maju", "juni": "junu", "juli": "julu", "avgust": "avgustu",
       "septembar": "septembru", "oktobar": "oktobru", "novembar": "novembru",
       "decembar": "decembru"}
LOC2NOM = {v: k for k, v in LOC.items()}
LOC2NOM["august"] = "avgust"  # bhas 2025 koristi engleski oblik
LOC2NOM["augustu"] = "avgust"
RIM = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
UNITS = {"kg", "l", "komad", "kutija", "par", "m3", "m2", "kwh", "t", "komplet",
         "porcija", "tretman", "pakiranje", "paket", "bočica", "set", "minuta",
         "uplata", "pregled", "karta", "stranica"}
DROP = {"mjesečna", "godišnja", "uplata", "feniranjem", "stranica"}
COMMA_DOCS = ("b_01", "b_04")


def norm(tag, ans):
    if tag[:4] in COMMA_DOCS and re.fullmatch(r"\d+\.\d+%?", ans):
        return ans.replace(".", ",")
    return ans


def must2(text, ans, ctx):
    if ans not in text and ans.replace(",", ".") not in text and ans.replace(".", ",") not in text:
        raise ValueError(f"odgovor nije u tekstu [{ctx}]: {ans!r}")


rows = []


def qa(img, q, a, text, ctx):
    tag = os.path.basename(img)[:4]
    a = norm(tag, a)
    must2(text, a, ctx)
    rows.append({"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
        {"role": "assistant", "content": [{"type": "text", "text": a}]}],
        "images": [img]})


def numbers_after(text, label, n=6):
    i = text.find(label)
    if i < 0:
        return []
    seg = text[i + len(label):i + len(label) + 120]
    return re.findall(r"\d{1,4}[.,]\d{1,2}%?|\d{2,4}[.,]\d", seg)[:n]


def cover_meta(t):
    m = re.search(r"SARAJEVO,\s*(\d{2}\.\d{2}\.\d{4})\.", t)
    date = m.group(1) if m else ""
    m = re.search(r"BROJ/NUMBER\s*(\d+)", t)
    broj = m.group(1) if m else ""
    mj, god = "", ""
    for cand in NOM + list(LOC.values()) + ["august", "augustu"]:
        m = re.search(r"u Bosni i Hercegovini,?\s*(?:u\s+)?" + cand + r"\s*(\d{4})", t)
        if m:
            mj, god = LOC2NOM.get(cand, cand), m.group(1)
            break
    return mj, god, broj, date


def cover_title(t, doc):
    pats = {"01": r"Indeks potrošačkih cijena u Bosni i Hercegovini, \w+ \d{4}",
            "02": r"Prosječne potrošačke cijene u Bosni i Hercegovini u \w+ \d{4}\. godine",
            "04": r"Indeks proizvođačkih cijena industrije u Bosni i Hercegovini u \w+ \d{4}\. godine"}
    m = re.search(pats[doc], t)
    return m.group(0) if m else ""


def main():
    import glob
    for pdf in sorted(glob.glob(f"{SRC}/PRI_*.pdf")):
        base = os.path.basename(pdf)  # PRI_01_2025_11_1_BS.pdf
        doc, god, mm = base[4:6], base[7:11], base[12:14]
        tag = f"{doc}_{god[-2:]}_{mm}"
        d = fitz.open(pdf)
        pages = [" ".join(p.get_text().split()) for p in d]
        t1 = pages[0]
        mj, god2, broj, date = cover_meta(t1)
        if not mj:
            print("SKIP (nema mjeseca):", base);
            continue
        mi = NOM.index(mj)
        rim, rim_prev, god_prev = RIM[mi], RIM[mi - 1], str(int(god) - 1)
        loc = LOC[mj]
        # render potrebnih stranica (naslovnica + tablice)
        keep = {"01": [1, 2], "02": [1, 2], "04": [1, 4]}
        for pg in keep[doc]:
            if pg <= len(d):
                d[pg - 1].get_pixmap(dpi=150).save(f"{IMG}/{tag}_p{pg}.png")
        img = lambda pg: f"data/bhas-bulk/img/{tag}_p{pg}.png"
        ctx0 = f"{doc}-{god}-{mm}"
        # naslovnica
        try:
            qa(img(1), "Koji je naslov dokumenta? Odgovori doslovno.",
               cover_title(t1, doc), t1, ctx0 + "-t")
            qa(img(1), "Kojeg datuma je objavljeno saopćenje? Odgovori samo datumom.",
               date, t1, ctx0 + "-d")
            qa(img(1), "Koji je broj saopćenja? Odgovori samo brojem.", broj, t1, ctx0 + "-b")
        except ValueError as e:
            print("SKIP naslovnica:", e)
        # narativ 01
        if doc == "01":
            for pat, q in [
                (r"u prosjeku su više za ([\d,]+%)",
                 f"Za koliko su u prosjeku više cijene u {loc} {god}. u odnosu na prethodni mjesec? Odgovori samo procentom."),
                (r"u prosjeku su niže za ([\d,]+%)",
                 f"Za koliko su u prosjeku niže cijene u {loc} {god}. u odnosu na prethodni mjesec? Odgovori samo procentom."),
                (r"isti mjesec prethodne godine viši je za ([\d,]+%)",
                 f"Za koliko su više cijene u {loc} {god}. u odnosu na isti mjesec prethodne godine? Odgovori samo procentom."),
                (r"Prijevoz za ([\d,]+%)",
                 "Za koliko su mjesečno porasle cijene prijevoza? Odgovori samo procentom.")]:
                m = re.search(pat, t1)
                if m:
                    qa(img(1), q, m.group(1), t1, ctx0 + "-n")
            t = pages[1]
            for label, q in [("Ukupan indeks", "ukupni indeks"),
                             ("Hrana i bezalkoholna pića", "indeks hrane i bezalkoholnih pića"),
                             ("Prijevoz", "indeks prijevoza"),
                             ("Odjeća i obuća", "indeks odjeće i obuće"),
                             ("Zdravstvo", "indeks zdravstva")]:
                v = numbers_after(t, label)
                if len(v) >= 2:
                    qa(img(2), f"Koliki je {q} za {rim} {god}. u odnosu na {rim_prev} {god}.? Odgovori samo brojem.",
                       v[0], t, ctx0 + "-t1")
                    qa(img(2), f"Koliki je {q} za {rim} {god}. u odnosu na {rim} {god_prev}.? Odgovori samo brojem.",
                       v[1], t, ctx0 + "-t2")
        # narativ 04
        if doc == "04":
            for pat, q in [
                (r"ukupnu industriju su zabilježile rast nivoa za ([\d,]+%), dok",
                 "Za koliko su mjesečno porasle proizvođačke cijene za ukupnu industriju? Odgovori samo procentom."),
                (r"isti mjesec prethodne godine zabilježile rast za ([\d,]+%)",
                 "Za koliko su godišnje porasle proizvođačke cijene za ukupnu industriju? Odgovori samo procentom.")]:
                m = re.search(pat, t1)
                if m:
                    qa(img(1), q, m.group(1), t1, ctx0 + "-n")
            if len(pages) >= 4:
                t = pages[3]
                for label, q, idx in [("UKUPAN INDEKS", f"ukupan indeks na domaćem tržištu za {rim} {god}. u odnosu na {rim_prev} {god}.", 0),
                                      ("UKUPAN INDEKS", f"ukupan indeks na ukupnom tržištu za {rim} {god}. u odnosu na {rim} {god_prev}.", 5),
                                      ("Energija", f"indeks energije na nedomaćem tržištu za {rim} {god}. u odnosu na {rim} {god_prev}.", 3)]:
                    v = numbers_after(t, label)
                    if len(v) > idx:
                        qa(img(4), f"Koliki je {q} Odgovori samo brojem.", v[idx], t, ctx0 + "-t")
        # cijene 02: sve s obje stranice
        if doc == "02":
            for pg in (1, 2):
                if pg > len(d):
                    continue
                d[pg - 1].get_pixmap(dpi=150).save(f"{IMG}/{tag}_p{pg}.png")
            dd = fitz.open(pdf)
            for pg in (1, 2):
                if pg > len(dd):
                    continue
                t = " ".join(dd[pg - 1].get_text().split())
                lines = t.split("  ")
                # linijski parse kao u make_bhas_train
                raw = dd[pg - 1].get_text().split("\n")
                for i, ln in enumerate(raw):
                    ln = ln.strip()
                    if ln in UNITS and i + 1 < len(raw) and re.fullmatch(r"\d+[.,]\d+", raw[i + 1].strip()) and i > 0:
                        lab = raw[i - 1].strip()
                        if lab and not re.fullmatch(r"[\d.,/% ]+", lab) and lab not in UNITS and len(lab) > 2 \
                                and lab.lower() not in ("mjesečna", "godišnja", "uplata", "feniranjem", "stranica"):
                            try:
                                qa(img(pg), f"Kolika je prosječna cijena ({lab.lower()})? Odgovori samo brojem.",
                                   raw[i + 1].strip(), t, ctx0 + f"-c{pg}")
                            except ValueError:
                                pass
    with open(f"{OUT}/train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"gotovo: uzoraka={len(rows)}")


if __name__ == "__main__":
    main()
