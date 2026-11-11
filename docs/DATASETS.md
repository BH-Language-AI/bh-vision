# Vision dataseti i benchmarkovi (BH-Vision Faza 0/1)

Datum: 2026-11-11. Ključni nalaz: **ne postoji javni OCR/dokument dataset za bosanski.**

## Kandidati (tehnike učiti ovdje — čiste licence)
- **PubLayNet** (IBM, ~336k str.) — anotacije CDLA-Permissive-1.0, slike PMC OA commercial-use. Layout pretraining.
- **DocLayNet** (IBM, 80k str.) — anotacije CDLA-Permissive-1.0; slike provjeriti po kategoriji.
- **PlotQA** — CC-BY-4.0 + MIT. Chart QA tehnika (bolja licenca od ChartQA).
- **XM3600** (Google) — CC-BY-4.0 anotacije; **hrvatski captioning DA**, bs/sr NE. Eval proxy za bs.
- **postocr-sr** (Sagicc) — CC-BY-4.0, srpska književnost. Post-OCR transfer na bs/hr latinicu.

## Samo istraživanje (NE komercijala bez pravnog mišljenja)
ChartQA (GPL-3.0), DocVQA (HF mirror ≠ prava slika), FUNSD (non-commercial), XFUND (CC-BY-NC-SA), RVL-CDIP (bez standalone licence), IAM (non-commercial + registracija), LAION-5B (slike pod copyrightom, PII/NSFW).

## Praznine (nema ničega javnog)
Bosanski OCR (čćdžšđ), bs forme/zakoni/tabele kao slike, chart QA na bs/hr/sr, mape, image-text instrukcije i video na bs/hr/sr, scene-text s dijakriticima (ICDAR MLT bez balkanskih).

## Preporuka
Tehnike na čistim engleskim (PubLayNet/PlotQA), eval proxy hr/sr (XM3600-hr, postocr-sr), za bosanski prikupiti vlastiti mali held-out set. Prije komercijalnog treninga čitati OBA sloja prava (anotacije + slike).
