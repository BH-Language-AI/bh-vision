# BH-Vision trening evidencija

## Smoke (2026-11-16) — PASS
- Kernel `admiragic/bh-vision-smoke-smo2b` v5, Tesla T4, SmolVLM2-2.2B QLoRA r8, 32 uzorka (v02), 20 koraka.
- Vrijeme: 1h22 (246 s/korak — sporo: puni-logit CE + ~1050 image tokena po slici).
- Final loss: 107.9. Adapter: 662 LoRA tenzora, 27 MB, verificiran (`safetensors` ključevi OK).
- Popravke putem: `num2words` dep → pin `transformers==5.22.2` (novije lome SmolVLM2 merge) →
  single-GPU (`CUDA_VISIBLE_DEVICES=0`, `device_map={"":0}`) → obični CE umjesto TRL fused kernela (sm<8.0).

## Prvi trening (2026-11-16, Kaggle) — napušten, zamijenjen LUMI v1 (ispod)
- Kernel `admiragic/bh-vision-train-smo2b-v1`, v03 (1962 train, split po slici), QLoRA r32/a64, lr 1.5e-4, 1 epoha (~61 korak, ~4 h).
- 1 epoha jer 2 ne staju u Kaggle limit uz 246 s/korak; nastavak po rezultatu.
- Cilj: loss pad + eval iznad baselinea 7/10 (netreniran 2.2B, v03-val, 10 uzoraka).

## Baseline (2026-11-16, CPU, netreniran 2.2B, v03-val 10 uzoraka): 7/10 normalized
- OCR iznosi savršeni; chart max/min griješi (august→Juli, 35→40, 40→30).

## LUMI v1 (2026-11-16/17) — završen, NEGATIVAN
- Trening `smo2b-v1` (frozen vision encoder, 2 epohe, 124 koraka, ~2 h na small-g): `TRAIN_DONE`, train_loss 33.31.
- Eval adaptera na cijelom valu: **26/217 = 12.0%** (`out/eval-smo2b-v1.json`, job 23819930) — debelo ispod baselinea → zamrznuti encoder + memorizacija, potvrđeno.
- Adapter (pravi PEFT: `adapter_config.json` + `adapter_model.safetensors`) ostaje na scratchu.

## Baseline full-val (2026-11-17, LUMI, netreniran 2.2B, v03-val 217 uzoraka): 81/217 = 37.3%
- (`out/eval-base-2b-v1.json`, job 23820476.) Uzorak od 10 (7/10) bio je sreća — pravi baseline je 37.3%.

## LUMI v2 (2026-11-17) — treniran, čeka eval
- `smo2b-v2` (ODLEĐEN vision encoder, 1 epoha, 62 koraka, ~1 h): `TRAIN_DONE`, train_loss 94.55.
- Pažnja: v2 `out/adapter-smo2b-v1` je PUNI model save (4.7 GB `model.safetensors`), ne PEFT — eval zato ide kao `--base`, skripta `slurm/bh-vision-eval-v2.sbatch` (uploadana u job-dir).
- Eval job predan 2026-11-18 ali odbijen: `QOSMaxSubmitJobPerUserLimit` (3 posla već aktivna: bh2-mixv + 2×amj-loop). Ponavljanje kad se oslobodi slot — NE gasiti tuđe poslove bez dogovora.

## Mini 500M (2026-11-17) — negativan rezultat
- Kernel `admiragic/bh-vision-mini-500m-v1`, T4, 24 koraka QLoRA r16 + eval 10 val uzoraka istim kernelom.
- Eval: **0/10 normalized** (eho prompta, halucinacije: `august→"Jednak je mjesec..."`, `10→"Špatičnost..."`).
- Netrenirani 2.2B baseline na istim tipovima: 7/10. Zaključak: 500M-video je preslab i/ili 24 koraka premalo — nije presuda za 2.2B put.
- Sljedeći GPU: Kaggle kvota se obnavlja 21.11. ili LUMI (treba EFP cert).

## Verzije (radeće, pinovano)
`transformers==5.22.2`, `trl==1.18.0`, `peft==0.21.2`. Novije od 5.22 lome SmolVLM2 merge (reshape `[-1,14,2048]`).
