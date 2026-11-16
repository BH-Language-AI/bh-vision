# BH-Vision trening evidencija

## Smoke (2026-11-16) — PASS
- Kernel `admiragic/bh-vision-smoke-smo2b` v5, Tesla T4, SmolVLM2-2.2B QLoRA r8, 32 uzorka (v02), 20 koraka.
- Vrijeme: 1h22 (246 s/korak — sporo: puni-logit CE + ~1050 image tokena po slici).
- Final loss: 107.9. Adapter: 662 LoRA tenzora, 27 MB, verificiran (`safetensors` ključevi OK).
- Popravke putem: `num2words` dep → pin `transformers==5.22.2` (novije lome SmolVLM2 merge) →
  single-GPU (`CUDA_VISIBLE_DEVICES=0`, `device_map={"":0}`) → obični CE umjesto TRL fused kernela (sm<8.0).

## Prvi trening (2026-11-16) — U TOKU
- Kernel `admiragic/bh-vision-train-smo2b-v1`, v03 (1962 train, split po slici), QLoRA r32/a64, lr 1.5e-4, 1 epoha (~61 korak, ~4 h).
- 1 epoha jer 2 ne staju u Kaggle limit uz 246 s/korak; nastavak po rezultatu.
- Cilj: loss pad + eval iznad baselinea 7/10 (netreniran 2.2B, v03-val, 10 uzoraka).

## Baseline (2026-11-16, CPU, netreniran 2.2B, v03-val 10 uzoraka): 7/10 normalized
- OCR iznosi savršeni; chart max/min griješi (august→Juli, 35→40, 40→30).

## Verzije (radeće, pinovano)
`transformers==5.22.2`, `trl==1.18.0`, `peft==0.21.2`. Novije od 5.22 lome SmolVLM2 merge (reshape `[-1,14,2048]`).
