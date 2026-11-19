"""BH held-out v01: PRAVI bosanski dokumenti (bhas.gov.ba saopcenja 09/2026).
Rucno verificirana QA — svaki par provjeren na slici. NIKAD u train.
Koristenje: SRC=/tmp/opencode/heldout-src IMG=data/heldout-v01/img OUT=data/heldout-v01 python3 scripts/make_heldout_bs.py
"""
import json
import os
import shutil

SRC = os.getenv("SRC", "/tmp/opencode/heldout-src")
IMG = os.getenv("IMG", "data/heldout-v01/img")
OUT = os.getenv("OUT", "data/heldout-v01")
os.makedirs(IMG, exist_ok=True)

# (src_png, pitanje, odgovor) — svi odgovori doslovno s dokumenta (zarez-decimale!)
QA = [
 # PRI_01 p1 — naslovnica + narativ
 ("PRI_01_p1.png", "Koji je naslov dokumenta? Odgovori doslovno.", "Indeks potrošačkih cijena u Bosni i Hercegovini, septembar 2026."),
 ("PRI_01_p1.png", "Kojeg datuma je objavljeno saopćenje? Odgovori samo datumom.", "27.10.2026."),
 ("PRI_01_p1.png", "Koji je broj saopćenja? Odgovori samo brojem.", "09"),
 ("PRI_01_p1.png", "Za koliko su u prosjeku više cijene u septembru 2026. u odnosu na prethodni mjesec? Odgovori samo procentom.", "1,2%"),
 ("PRI_01_p1.png", "Za koliko su više cijene u septembru 2026. u odnosu na isti mjesec prethodne godine? Odgovori samo procentom.", "6,0%"),
 ("PRI_01_p1.png", "Za koliko su mjesečno porasle cijene prijevoza? Odgovori samo procentom.", "3,4%"),
 ("PRI_01_p1.png", "Za koliko su na godišnjem nivou pale cijene odjeće i obuće? Odgovori samo procentom.", "6,9%"),
 ("PRI_01_p1.png", "Za koliko su godišnje porasle cijene stanovanja i režijskih izdataka? Odgovori samo procentom.", "12,7%"),
 # PRI_01 p2 — CPI tabela
 ("PRI_01_p2.png", "Koji je naslov tabele? Odgovori doslovno.", "Indeks potrošačkih cijena prema klasifikaciji COICOP u septembru 2026. godine"),
 ("PRI_01_p2.png", "Koliki je ukupan indeks za IX 2026. u odnosu na VIII 2026.? Odgovori samo brojem.", "101,2"),
 ("PRI_01_p2.png", "Koliki je indeks hrane i bezalkoholnih pića za IX 2026. u odnosu na VIII 2026.? Odgovori samo brojem.", "100,6"),
 ("PRI_01_p2.png", "Koliki je indeks prijevoza za IX 2026. u odnosu na IX 2025.? Odgovori samo brojem.", "127,6"),
 ("PRI_01_p2.png", "Koliki je indeks zdravstva za IX 2026. u odnosu na Ø2015? Odgovori samo brojem.", "130,6"),
 ("PRI_01_p2.png", "Koliki je indeks obuće za IX 2026. u odnosu na VIII 2026.? Odgovori samo brojem.", "102,3"),
 ("PRI_01_p2.png", "Koliki je indeks duhana za IX 2026. u odnosu na IX 2025.? Odgovori samo brojem.", "104,6"),
 ("PRI_01_p2.png", "Koliki je indeks odjeće i obuće za IX 2026. u odnosu na IX 2025.? Odgovori samo brojem.", "93,1"),
 ("PRI_01_p2.png", "Koliki je indeks stanovanja za I-IX 2026. u odnosu na I-IX 2025.? Odgovori samo brojem.", "110,3"),
 # PRI_02 p1 — naslovnica + napomena + tablica 1. dio
 ("PRI_02_p1.png", "Koji je naslov dokumenta? Odgovori doslovno.", "Prosječne potrošačke cijene u Bosni i Hercegovini u septembru 2026. godine"),
 ("PRI_02_p1.png", "Ko objavljuje relevantne indekse potrošačkih cijena za nivo BiH? Odgovori samo nazivom.", "Agencija za statistiku BiH"),
 ("PRI_02_p1.png", "Kolika je prosječna cijena riže po kilogramu? Odgovori samo brojem.", "4.4"),
 ("PRI_02_p1.png", "Kolika je prosječna cijena junetine bez kosti po kilogramu? Odgovori samo brojem.", "27.0"),
 ("PRI_02_p1.png", "Kolika je prosječna cijena kokošijih jaja po komadu? Odgovori samo brojem.", "0.4"),
 ("PRI_02_p1.png", "Kolika je prosječna cijena mlijeka po litru? Odgovori samo brojem.", "1.7"),
 ("PRI_02_p1.png", "Kolika je prosječna cijena meda po kilogramu? Odgovori samo brojem.", "23.0"),
 ("PRI_02_p1.png", "Kolika je prosječna cijena muških kožnih cipela po paru? Odgovori samo brojem.", "175.5"),
 ("PRI_02_p1.png", "Kolika je prosječna cijena vode po kubnom metru? Odgovori samo brojem.", "1.4"),
 ("PRI_02_p1.png", "Kolika je prosječna cijena mljevene kahve po kilogramu? Odgovori samo brojem.", "28.4"),
 ("PRI_02_p1.png", "Kolika je prosječna cijena putera po kilogramu? Odgovori samo brojem.", "31.3"),
 # PRI_02 p2 — tablica 2. dio
 ("PRI_02_p2.png", "Kolika je prosječna cijena kanalizacije po kubnom metru? Odgovori samo brojem.", "0.7"),
 ("PRI_02_p2.png", "Kolika je prosječna cijena mrkog uglja po toni? Odgovori samo brojem.", "274.8"),
 ("PRI_02_p2.png", "Kolika je prosječna cijena frižidera sa zamrzivačem po komadu? Odgovori samo brojem.", "678.3"),
 ("PRI_02_p2.png", "Kolika je prosječna cijena euro dizela po litru? Odgovori samo brojem.", "3.6"),
 ("PRI_02_p2.png", "Kolika je mjesečna uplata za vrtić? Odgovori samo brojem.", "291.1"),
 ("PRI_02_p2.png", "Kolika je cijena šnicle po porciji? Odgovori samo brojem.", "21.3"),
 ("PRI_02_p2.png", "Kolika je cijena fotokopiranja po stranici? Odgovori samo brojem.", "0.2"),
 ("PRI_02_p2.png", "Kolika je godišnja uplata za obavezno osiguranje motornih vozila? Odgovori samo brojem.", "376.6"),
 ("PRI_02_p2.png", "Kolika je cijena pice po komadu? Odgovori samo brojem.", "13.2"),
 ("PRI_02_p2.png", "Kolika je cijena tehničkog pregleda vozila? Odgovori samo brojem.", "74.0"),
 # PRI_04 p1 — naslovnica + narativ
 ("PRI_04_p1.png", "Koji je naslov dokumenta? Odgovori doslovno.", "Indeks proizvođačkih cijena industrije u Bosni i Hercegovini u septembru 2026. godine"),
 ("PRI_04_p1.png", "Koja godina izdanja je navedena u zaglavlju? Odgovori samo oznakom.", "XVI"),
 ("PRI_04_p1.png", "Za koliko su mjesečno porasle proizvođačke cijene za ukupnu industriju? Odgovori samo procentom.", "0,2%"),
 ("PRI_04_p1.png", "Za koliko su godišnje porasle proizvođačke cijene za ukupnu industriju? Odgovori samo procentom.", "5,5%"),
 ("PRI_04_p1.png", "Za koliko su mjesečno porasle cijene na domaćem tržištu? Odgovori samo procentom.", "0,1%"),
 ("PRI_04_p1.png", "Za koliko su godišnje porasle cijene na domaćem tržištu? Odgovori samo procentom.", "4,7%"),
 ("PRI_04_p1.png", "Za koliko su mjesečno porasle cijene na nedomaćem tržištu? Odgovori samo procentom.", "0,4%"),
 ("PRI_04_p1.png", "Za koliko su godišnje porasle cijene na nedomaćem tržištu? Odgovori samo procentom.", "6,7%"),
 ("PRI_04_p1.png", "Za koliko je zabilježen pad kod trajnih proizvoda? Odgovori samo procentom.", "0,2%"),
 ("PRI_04_p1.png", "Za koliko su godišnje porasle cijene energije? Odgovori samo procentom.", "7,7%"),
 # PRI_04 p2 — narativ domace trziste
 ("PRI_04_p2.png", "Za koliko su mjesečno porasle cijene energije na domaćem tržištu? Odgovori samo procentom.", "0,3%"),
 ("PRI_04_p2.png", "Za koliko su mjesečno porasle cijene kapitalnih proizvoda na domaćem tržištu? Odgovori samo procentom.", "0,8%"),
 ("PRI_04_p2.png", "Za koliko su godišnje porasle cijene intermedijarnih proizvoda na domaćem tržištu? Odgovori samo procentom.", "6,7%"),
 # PRI_04 p3 — narativ nedomace trziste
 ("PRI_04_p3.png", "Za koliko su mjesečno porasle cijene vađenja ruda i kamena na nedomaćem tržištu? Odgovori samo procentom.", "0,2%"),
 ("PRI_04_p3.png", "Za koliko su godišnje porasle cijene prerađivačke industrije na nedomaćem tržištu? Odgovori samo procentom.", "6,9%"),
 # PRI_04 p4 — PPI tabela (vizualno verificirano)
 ("PRI_04_p4.png", "Koliki je ukupan indeks na domaćem tržištu za IX 2026. u odnosu na VIII 2026.? Odgovori samo brojem.", "100,1"),
 ("PRI_04_p4.png", "Koliki je ukupan indeks na ukupnom tržištu za IX 2026. u odnosu na IX 2025.? Odgovori samo brojem.", "105,5"),
 ("PRI_04_p4.png", "Koliki je indeks energije na nedomaćem tržištu za IX 2026. u odnosu na IX 2025.? Odgovori samo brojem.", "112,7"),
 ("PRI_04_p4.png", "Koliki je indeks proizvodnje koksa na ukupnom tržištu za IX 2026. u odnosu na IX 2025.? Odgovori samo brojem.", "129,3"),
 ("PRI_04_p4.png", "Koliki je indeks proizvodnje koksa na nedomaćem tržištu za IX 2026. u odnosu na IX 2025.? Odgovori samo brojem.", "183,9"),
 ("PRI_04_p4.png", "Koliki je indeks kapitalnih proizvoda na domaćem tržištu za IX 2026. u odnosu na VIII 2026.? Odgovori samo brojem.", "100,8"),
 ("PRI_04_p4.png", "Koliki je indeks vađenja ugljena na domaćem tržištu za IX 2026. u odnosu na IX 2025.? Odgovori samo brojem.", "103,5"),
 ("PRI_04_p4.png", "Koliki je indeks proizvodnje odjeće na ukupnom tržištu za IX 2026. u odnosu na VIII 2026.? Odgovori samo brojem.", "104,0"),
 # PRI_04 p6 — mjesecne tablice (vizualno verificirano)
 ("PRI_04_p6.png", "Koliki je ukupan indeks za septembar 2026. (prosjek 2021=100)? Odgovori samo brojem.", "135,4"),
 ("PRI_04_p6.png", "Koliki je indeks domaćeg tržišta za septembar 2026. (prosjek 2021=100)? Odgovori samo brojem.", "139,1"),
 ("PRI_04_p6.png", "Koliki je indeks nedomaćeg tržišta za septembar 2026. (prosjek 2021=100)? Odgovori samo brojem.", "131,5"),
 ("PRI_04_p6.png", "Koliki je ukupan indeks za septembar 2026. u odnosu na prethodni mjesec? Odgovori samo brojem.", "100,2"),
 ("PRI_04_p6.png", "Koliki je indeks domaćeg tržišta za septembar 2026. u odnosu na isti mjesec prethodne godine? Odgovori samo brojem.", "104,7"),
 # PRI_04 p7 — graf (vizualno verificirano)
 ("PRI_04_p7.png", "Koji je naslov grafa? Odgovori doslovno.", "Godišnje stope promjene cijena proizvođača u BiH po tipu tržišta, septembar 2025.- septembar 2026."),
 ("PRI_04_p7.png", "Koliko linija ima na grafu? Odgovori samo brojem.", "3"),
 ("PRI_04_p7.png", "Koje tržište ima najvišu godišnju stopu u septembru 2026.? Odgovori samo nazivom.", "Nedomaće tržište"),
 ("PRI_04_p7.png", "Koje tržište prikazuje plava linija? Odgovori samo nazivom.", "Ukupno tržište"),
 ("PRI_04_p7.png", "Koje tržište ima najnižu stopu u septembru 2025.? Odgovori samo nazivom.", "Nedomaće tržište"),
 # izdavaci
 ("PRI_02_p3.png", "Ko je izdavač publikacije? Odgovori samo nazivom.", "Agencija za statistiku Bosne i Hercegovine"),
 ("PRI_04_p10.png", "Ko je izdavač publikacije? Odgovori samo nazivom.", "Agencija za statistiku Bosne i Hercegovine"),
]


def main():
    rows = []
    for i, (png, q, a) in enumerate(QA):
        src = f"{SRC}/img/{png}"
        dst = f"{IMG}/ho{i:04d}.png"
        shutil.copy(src, dst)
        rows.append({"messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
            {"role": "assistant", "content": [{"type": "text", "text": a}]}],
            "images": [dst.replace("data/", "data/")]})
    # relativne putanje od repo root-a
    for r in rows:
        r["images"] = [p if p.startswith("data/") else p for p in r["images"]]
    with open(f"{OUT}/eval.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"gotovo: uzoraka={len(rows)}")


if __name__ == "__main__":
    main()
