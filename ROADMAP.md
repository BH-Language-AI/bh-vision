# BH-Vision — Roadmap (stanje 2026-11-18, commit bd55049)

- [x] Faza 0/1: istraživanje, arhitektonske odluke, repo temelji
- [x] Faza 2 (djelimično): pilot SmolVLM2-2.2B — smoke PASS (T4, 20 koraka), baseline 7/10 na v03-val/10 uzoraka; mini 500M negativan 0/10. 8B/12B nisu pilotirani.
- [x] Faza 3 (djelimično): registar + vlastiti synth v03 (1962 train / 217 val, split po slici); held-out `bs-vision-ocr-heldout-v01` još prazan placeholder; proxy XM3600-hr + postocr-sr.
- [ ] Faza 4: compute — Kaggle korišten (kvota se obnavlja 21.11); LUMI sbatch skripte spremne (`slurm/`), čeka EFP cert (Shaova strana).
- [x] Faza 5 (djelimično): `training/`, `slurm/`, `kaggle-*/`, `configs/`, `scripts/` postoje; pravi `evaluation/`/`benchmarks/` još ne.
- [ ] Faza 6: inference prototip — `training/eval.py` radi baseline na CPU (10 uzoraka); full-val baseline sbatch spreman, nije pokrenut.
- [ ] Faza 7: trening — v2 recept spreman (`configs/smo2b-v2-unfrozen.yaml`, CONFIG arg); sljedeći GPU slot + odobrenje. Prvi smo2b trening bio u toku na Kaggleu.
- [ ] Faza 8: release — kartice placeholderi bez težina (kolekcija live); objava tek nakon odobrenog treninga i verificirane evaluacije.
