![BH-Vision banner](assets/banner-bh-vision.png)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![HF Collection](https://img.shields.io/badge/🤗_Collection-BH--Vision-yellow.svg)](https://huggingface.co/collections/BH-Language-AI/bh-vision)
[![Faza](https://img.shields.io/badge/faza-0%2F1_istraživanje-lightgrey.svg)](ROADMAP.md)
[![CI](https://github.com/BH-Language-AI/bh-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/BH-Language-AI/bh-vision/actions)

# 👁️ BH-Vision

**Vizuelni AI za bosanski jezik i Balkan — slike, dokumenti, OCR, chartovi, video.**
Prvi bosanski vision-language projekat: adaptacija otvorenih VLM-ova (SmolVLM2, Idefics3, Pixtral) na bosanski — sa čćdžšđ, ijekavicom i domaćim dokumentima.

> **Status: Faza 0/1 — istraživanje i temelji.** Model kartice su placeholderi bez težina; težine se objavljuju tek nakon odobrenog treninga i verificirane evaluacije.

## 🗺️ Gdje je šta

| Šta | Gdje |
|---|---|
| 🤗 Modeli (`bh-vision-1-2b/8b/12b`) | [HF kolekcija BH-Vision](https://huggingface.co/collections/BH-Language-AI/bh-vision) |
| 📦 Trening set (`bs-vision-train-v03`, 2179 uzoraka) | [HF dataset](https://huggingface.co/datasets/BH-Language-AI/bs-vision-train-v03) |
| 🧪 Held-out eval (prazan placeholder) | [HF dataset](https://huggingface.co/datasets/BH-Language-AI/bs-vision-ocr-heldout-v01) |
| 📖 Odluke, registar, roadmap | [BH-LANGUAGE-AI](https://github.com/BH-Language-AI) (interni plan) |

## 🚀 Brzi start

```bash
pip install -e ".[train]"   # torch, transformers==5.22.2, trl, peft (pinovano!)
make doctor                 # suha provjera
python scripts/make_synth_bs.py              # regeneriši sintetički set
python training/train.py --config configs/smo2b-qlora.yaml --train data/synth3/train.jsonl --out releases/adapter-v01 --quant none
```

> ⚠️ Trening troši GPU — samo uz odobrenje. Prvi smoke (T4, 20 koraka) prošao; evidencija u [`docs/TRAINING.md`](docs/TRAINING.md).

## 📚 Dokumenti

- `docs/PROJECT_CHARTER.md` — misija, jezici (bs primarno, ijekavica), moduli
- `docs/ARCHITECTURE.md` — Arch A (adaptacija postojećeg VLM-a), 2B + 8B (+12B opcionalno), vision adapter odvojen od tekstualnog frameworka
- `docs/MODELS.md` — EU kandidati s licencama
- `docs/DATASETS.md` — registar + praznine za bosanski + vlastiti setovi (v03 aktuelan)
- `docs/TRAINING.md` — smoke rezultati, baseline 7/10, pinovane verzije
- `slurm/` — LUMI job skripte (`dev-g`, MI250X, bf16)
- `kaggle-*/` — Kaggle kereneli (smoke, trening, eval)

## 🧭 Ključne odluke

1. Fiksne 2B/4B/7B nisu opravdane — cilj: **2B (SmolVLM2)** + **8B (Idefics3)**, 12B (Pixtral) opcionalno.
2. Kratkoročno samo adaptacija postojećeg VLM-a; spajanje na BH-LLM-Basic tek kad backbone postoji.
3. Bosanski OCR ne postoji javno — vlastiti sintetički trening + held-out eval; proxy XM3600-hr + postocr-sr.
4. `transformers==5.22.2` pin — novije verzije lome SmolVLM2 merge.

## 🤝 Kontakt

- E-mail: xnetba@gmail.com
- Organizacija: [BH-Language-AI](https://github.com/BH-Language-AI) (Kolektiv d.o.o. × Gamatrade d.o.o.)
- Pravila: vanjski resursi se referenciraju, ne kopiraju; objave težina/podataka samo uz odobrenje.
