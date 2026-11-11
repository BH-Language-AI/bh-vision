# BH-Vision — Faza 0/1: nalazi i odluke

Datum: 2026-11-11. Detalji: `reports/research/2026-11-11-vision-models.md`, `2026-11-11-vision-datasets.md`.

## Modeli (svi Apache-2.0 osim navedenog; bs/hr/sr NIGDJE dokumentiran)
- Najlakši: **SmolVLM2-256M/500M** (1.4–5GB VRAM, video+slika)
- Generalni: **SmolVLM2-2.2B-Instruct** (DocVQA ~80, consumer GPU)
- Napredni: **Pixtral-12B-2409** (128k ctx, ChartQA 81.8, ~24GB+ VRAM, bez Mistral komercijalnih plafona)
- Integracijski: **Idefics3-8B-Llama3** (Llama-3.1 backbone; napomena: roditeljska Meta licenca — pravna provjera prije produkcije)
- Salamandra-VL-7B: SAMO istraživačko poređenje (RAIL research-only, diskvalificiran za proizvod)
- Baseline: Qwen2.5-VL-7B (Apache-2.0), Gemma-3 (click-through licenca)

## Odluke
1. **Veličine 2B/4B/7B nisu tehnički opravdane kao fiksne.** Nema EU 4B kandidata; 7B klasa je realno 8B (Idefics3) ili 12B (Pixtral). Prijedlog: ciljati 2B (SmolVLM2) + 8B (Idefics3) + opcionalno 12B (Pixtral); oznake 4B/7B ne koristiti dok ne postoji checkpoint.
2. **Arhitektura A (adaptacija postojećeg VLM-a) je jedini kratkoročni put.** Arhitektura B (encoder na BH-LLM-Basic) je teoretska — backbone modeli ne postoje.
3. **AgentMujo Training je tekst-only** (audit Faze 0 BH-LLM-Basic) → za vision treba odvojeni adapter/modul, bez diranja jezičkog dijela.
4. **Bosanski OCR/dokumenti ne postoje javno** → tehnike na PubLayNet/PlotQA, proxy eval XM3600-hr + postocr-sr, vlastiti held-out set za bs.
5. Ništa se ne trenira, ne objavljuje (težine), ne redistribuira bez odobrenja i provjere prava.
