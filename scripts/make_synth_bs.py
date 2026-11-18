"""Sinteticki bs trening set v02: dokumenti (PIL) + chartovi (matplotlib), 100% vlastiti sadrzaj, CC-BY-4.0.
OUT=data/synth2 N_DOC=800 N_CHART=400 prośliruje v01 (novi tipovi: faktura-tabela, izvod, line-chart, vise QA).
Sadrzi dijakritike ccdzsdj; held-out/proxy setovi se NIKAD ne mijesaju ovdje.
"""
import json, os, random
random.seed(int(os.getenv("SEED", "43")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

OUT = os.getenv("OUT", "data/synth2")
IMG = f"{OUT}/img"
N_DOC, N_CHART = int(os.getenv("N_DOC", "800")), int(os.getenv("N_CHART", "400"))
os.makedirs(IMG, exist_ok=True)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
TITLE = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
MIJESECI = ["januar", "februar", "mart", "april", "maj", "juni",
            "juli", "august", "septembar", "oktobar", "novembar", "decembar"]
ARTIKLI = ["Hljeb", "Mlijeko", "Kafa", "Šećer", "Čaj", "Džem", "Sir", "Brašno",
           "Ulje", "Jaja", "Čokolada", "Džus", "Šampon", "Sapun", "Žvake", "Ćevapi"]
IMENA = ["Šefik Džeko", "Amela Čaušević", "Tarik Hodžić", "Lejla Šabić",
         "Dženan Ćosić", "Aida Žunić", "Haris Bećirović", "Selma Mujdžić"]

rows = []


def save_aug(img, p):
    if random.random() < 0.3:  # lagana rotacija za robusnost OCR-a
        img = img.rotate(random.uniform(-1.5, 1.5), resample=Image.BICUBIC, fillcolor="white")
    img.save(p, quality=72)


def qa(img_p, q, a):
    rows.append({"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
        {"role": "assistant", "content": [{"type": "text", "text": a}]}],
        "images": [img_p]})


def doc_racun(i):
    stavke = random.sample(ARTIKLI, random.randint(2, 5))
    cijene = [round(random.uniform(1, 30), 2) for _ in stavke]
    uk = round(sum(cijene) * 1.17, 2)
    br = 2000 + i
    lines = [f"RAČUN br. {br}", "Prodavnica DŽAJIĆ", "Šeherdžikova 12, Sarajevo", ""]
    for a, c in zip(stavke, cijene):
        lines.append(f"{a} — {c:.2f} KM")
    lines += ["", "PDV 17% uračunat", f"UKUPNO: {uk:.2f} KM"]
    img = Image.new("RGB", (800, 120 + 55 * len(lines)), "white")
    d = ImageDraw.Draw(img)
    y = 30
    for j, ln in enumerate(lines):
        d.text((40, y), ln, font=TITLE if j == 0 else FONT, fill="black")
        y += 55
    p = f"{IMG}/doc{i:04d}.jpg"
    save_aug(img, p)
    qa(p, "Koliki je ukupan iznos na računu? Odgovori samo iznosom.", f"{uk:.2f} KM")
    if random.random() < 0.5:
        qa(p, "Koji je broj računa? Odgovori samo brojem.", str(br))
    if random.random() < 0.4:
        qa(p, "Koliko stavki ima račun? Odgovori samo brojem.", str(len(stavke)))


def doc_faktura(i):
    stavke = random.sample(ARTIKLI, random.randint(3, 5))
    cijene = [round(random.uniform(2, 40), 2) for _ in stavke]
    br = 5000 + i
    lines = [f"FAKTURA br. {br}", "Dobavljač: ŽITOPROMET", "Čekrkčića 5, Tuzla", ""]
    for a, c in zip(stavke, cijene):
        lines.append(f"{a} | 1 kom | {c:.2f} KM")
    img = Image.new("RGB", (800, 120 + 55 * len(lines)), "white")
    d = ImageDraw.Draw(img)
    y = 30
    for j, ln in enumerate(lines):
        d.text((40, y), ln, font=TITLE if j == 0 else FONT, fill="black")
        y += 55
    p = f"{IMG}/doc{i:04d}.jpg"
    save_aug(img, p)
    k = random.randrange(len(stavke))
    qa(p, f"Kolika je cijena za {stavke[k]}? Odgovori samo iznosom.", f"{cijene[k]:.2f} KM")
    qa(p, "Koliko stavki ima faktura? Odgovori samo brojem.", str(len(stavke)))


def doc_obavijest(i):
    naslov = random.choice(["OBAVJEŠTENJE", "POTVRDA", "OPOMENA", "RJEŠENJE"])
    ime = random.choice(IMENA)
    salter = random.randint(1, 6)
    lines = [naslov, f"Građanin: {ime}", f"Šalter {salter}, prizemlje", "Ponesite ličnu kartu i dokaz o uplati."]
    img = Image.new("RGB", (800, 420), "white")
    d = ImageDraw.Draw(img)
    y = 30
    for j, ln in enumerate(lines):
        d.text((40, y), ln, font=TITLE if j == 0 else FONT, fill="black")
        y += 60
    p = f"{IMG}/doc{i:04d}.jpg"
    save_aug(img, p)
    qa(p, "Prepiši naslov dokumenta.", naslov)
    if random.random() < 0.5:
        qa(p, "Kako se zove građanin? Odgovori samo imenom.", ime)


def doc_izvod(i):
    stanje = round(random.uniform(100, 9000), 2)
    br = 700000 + i
    lines = ["IZVOD br. " + str(br), "Banka: ŠTEDIONICA", f"Račun: {br}", "",
             f"Stanje: {stanje:.2f} KM", "Dostupno odmah"]
    img = Image.new("RGB", (800, 480), "white")
    d = ImageDraw.Draw(img)
    y = 30
    for j, ln in enumerate(lines):
        d.text((40, y), ln, font=TITLE if j == 0 else FONT, fill="black")
        y += 60
    p = f"{IMG}/doc{i:04d}.jpg"
    save_aug(img, p)
    qa(p, "Koliko je stanje računa? Odgovori samo iznosom.", f"{stanje:.2f} KM")


def chart_bar(i, months, vals, tag):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(months, vals)
    ax.set_ylabel("Prodaja (KM)")
    ax.set_title(f"Prodaja po mjesecima — {2024 + (i % 3)}.")
    p = f"{IMG}/{tag}{i:04d}.jpg"
    fig.savefig(p)
    plt.close(fig)
    best = months[vals.index(max(vals))]
    qa(p, "Koji mjesec ima najveću prodaju? Odgovori samo imenom mjeseca.", best)
    m = random.choice(months)
    qa(p, f"Kolika je prodaja za {m}? Odgovori samo brojem.", str(vals[months.index(m)]))
    if random.random() < 0.5:
        qa(p, "Kolika je razlika između najveće i najmanje prodaje? Odgovori samo brojem.",
           str(max(vals) - min(vals)))
    if random.random() < 0.5:  # v04: ciljano gusce — min mjesec + jos jedna vrijednost
        worst = months[vals.index(min(vals))]
        qa(p, "Koji mjesec ima najmanju prodaju? Odgovori samo imenom mjeseca.", worst)
        m2 = random.choice(months)
        qa(p, f"Kolika je prodaja za {m2}? Odgovori samo brojem.", str(vals[months.index(m2)]))


def chart_line(i, months, vals, tag):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(months, vals, marker="o")
    ax.set_ylabel("Temperatura (°C)")
    ax.set_title(f"Srednja temperatura — {2024 + (i % 3)}.")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    p = f"{IMG}/{tag}{i:04d}.jpg"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    worst = months[vals.index(min(vals))]
    qa(p, "Koji mjesec je najhladniji? Odgovori samo imenom mjeseca.", worst)
    m = random.choice(months)
    qa(p, f"Kolika je temperatura za {m}? Odgovori samo brojem.", str(vals[months.index(m)]))
    if random.random() < 0.5:  # v04: najtopliji + jos jedna vrijednost
        best = months[vals.index(max(vals))]
        qa(p, "Koji mjesec je najtopliji? Odgovori samo imenom mjeseca.", best)
        m2 = random.choice(months)
        qa(p, f"Kolika je temperatura za {m2}? Odgovori samo brojem.", str(vals[months.index(m2)]))


fns = [doc_racun, doc_faktura, doc_obavijest, doc_izvod]
for i in range(N_DOC):
    fns[i % 4](i)
for i in range(N_DOC, N_DOC + N_CHART):
    months = random.sample(MIJESECI, 6)
    vals = [random.randint(5, 60) for _ in months]
    if i % 2 == 0:
        chart_bar(i, months, vals, "chart")
    else:
        chart_line(i, months, [random.randint(-5, 30) for _ in months], "chart")

from collections import defaultdict
groups = defaultdict(list)
for r in rows:
    groups[r["images"][0]].append(r)
glist = list(groups.values())
random.shuffle(glist)
k = int(len(glist) * 0.9)
train_g, val_g = glist[:k], glist[k:]
for name, part in (("train", train_g), ("val", val_g)):
    flat = [r for g in part for r in g]
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in flat:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
n_tr = sum(len(g) for g in train_g); n_va = sum(len(g) for g in val_g)
print(f"gotovo: train={n_tr} val={n_va} slika={len(groups)} (split po slici, bez curenja)")
