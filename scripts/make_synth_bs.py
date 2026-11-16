"""Sinteticki bs trening set: dokumenti (PIL) + chartovi (matplotlib), 100% vlastiti sadrzaj, CC-BY-4.0.
Sadrzi dijakritike ccdzsdj; held-out/proxy setovi se NIKAD ne mijesaju ovdje.
Izlaz: data/synth/{train,val}.jsonl + data/synth/img/*.jpg (split 90/10, seed 42).
"""
import json, os, random
random.seed(42)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

OUT, IMG = "data/synth", "data/synth/img"
os.makedirs(IMG, exist_ok=True)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
TITLE = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
MIJESECI = ["januar", "februar", "mart", "april", "maj", "juni"]
ARTIKLI = ["Hljeb", "Mlijeko", "Kafa", "Šećer", "Čaj", "Džem", "Sir", "Brašno"]

rows = []


def doc_racun(i):
    stavke = random.sample(ARTIKLI, 3)
    cijene = [round(random.uniform(1, 20), 2) for _ in stavke]
    uk = round(sum(cijene) * 1.17, 2)
    br = 1000 + i
    lines = [f"RAČUN br. {br}", "Prodavnica DŽAJIĆ", "Šeherdžikova 12, Sarajevo", ""]
    for a, c in zip(stavke, cijene):
        lines.append(f"{a} — {c:.2f} KM")
    lines += ["", f"PDV 17% uračunat", f"UKUPNO: {uk:.2f} KM"]
    img = Image.new("RGB", (800, 600), "white")
    d = ImageDraw.Draw(img)
    y = 30
    for j, ln in enumerate(lines):
        d.text((40, y), ln, font=TITLE if j == 0 else FONT, fill="black")
        y += 55
    p = f"{IMG}/doc{i:04d}.jpg"
    img.save(p, quality=75)
    gold = f"{uk:.2f} KM"
    rows.append({"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Koliki je ukupan iznos na računu? Odgovori samo iznosom."}]},
        {"role": "assistant", "content": [{"type": "text", "text": gold}]}],
        "images": [p]})


def doc_obavijest(i):
    naslov = random.choice(["OBAVJEŠTENJE", "POTVRDA", "OPOMENA"])
    ime = random.choice(["Šefik Džeko", "Amela Čaušević", "Tarik Hodžić", "Lejla Šabić"])
    lines = [naslov, f"Građanin: {ime}", "Šalter 3, prizemlje", "Ponesite ličnu kartu i dokaz o uplati."]
    img = Image.new("RGB", (800, 500), "white")
    d = ImageDraw.Draw(img)
    y = 30
    for j, ln in enumerate(lines):
        d.text((40, y), ln, font=TITLE if j == 0 else FONT, fill="black")
        y += 60
    p = f"{IMG}/doc{i:04d}.jpg"
    img.save(p, quality=75)
    rows.append({"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Prepiši naslov dokumenta."}]},
        {"role": "assistant", "content": [{"type": "text", "text": naslov}]}],
        "images": [p]})


def chart(i):
    vals = [random.randint(5, 50) for _ in MIJESECI]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(MIJESECI, vals)
    ax.set_ylabel("Prodaja (KM)")
    ax.set_title(f"Prodaja po mjesecima — {2024 + (i % 3)}.")
    p = f"{IMG}/chart{i:04d}.jpg"
    fig.savefig(p)
    plt.close(fig)
    best = MIJESECI[vals.index(max(vals))]
    rows.append({"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Koji mjesec ima najveću prodaju? Odgovori samo imenom mjeseca."}]},
        {"role": "assistant", "content": [{"type": "text", "text": best}]}],
        "images": [p]})
    m = random.choice(MIJESECI)
    rows.append({"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Kolika je prodaja za {m}? Odgovori samo brojem."}]},
        {"role": "assistant", "content": [{"type": "text", "text": str(vals[MIJESECI.index(m)])}]}],
        "images": [p]})


for i in range(200):
    (doc_racun if i % 2 == 0 else doc_obavijest)(i)
for i in range(200, 300):
    chart(i)
random.shuffle(rows)
k = int(len(rows) * 0.9)
for name, part in (("train", rows[:k]), ("val", rows[k:])):
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"gotovo: train={len(rows[:k])} val={len(rows[k:])} slika={len(set(sum([r['images'] for r in rows], [])))}")
