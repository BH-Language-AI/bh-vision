# BH-Vision

Bosnian and Balkan Vision Intelligence — slike, dokumenti, OCR, video.
**Status: Faza 0/1 — istraživanje i temelji.** Težine se ne objavljuju bez odobrenja.

- Kolekcija: https://huggingface.co/collections/BH-Language-AI/bh-vision
- Kontakt: xnetba@gmail.com

## Dokumenti
- `docs/PROJECT_CHARTER.md` — misija, jezici (bs primarno, ijekavica), moduli
- `docs/ARCHITECTURE.md` — odluke: Arch A (adaptacija), veličine 2B/8B(/12B), vision adapter odvojen od tekstualnog frameworka
- `docs/MODELS.md` — EU kandidati s licencama (izvor: izvještaj 2026-11-11-vision-models)
- `docs/DATASETS.md` — registar + praznine za bosanski

## Ključne odluke Faze 0/1
1. Fiksne 2B/4B/7B nisu opravdane — cilj: 2B (SmolVLM2) + 8B (Idefics3), 12B (Pixtral) opcionalno.
2. Kratkoročno samo adaptacija postojećeg VLM-a; spajanje na BH-LLM-Basic tek kad backbone postoji.
3. Bosanski OCR ne postoji javno — treba vlastiti eval set.
