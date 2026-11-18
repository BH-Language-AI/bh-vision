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

## LUMI v2 (2026-11-17/18) — završen, KATASTROFALAN
- `smo2b-v2` (ODLEĐEN vision encoder, 1 epoha, 62 koraka, ~1 h): `TRAIN_DONE`, train_loss 94.55 (vs v1 33.31 — znak divergencije).
- Eval na cijelom valu: **0/217 = 0.0%** (`out/eval-smo2b-v2.json`, job 23840606) — model kolapsirao u repetitivno brbljanje (`45.41 KM` → `Josiah Josiah Josiah...`).
- Zaključak: i v1 (12.0%) i v2 (0.0%) degradiraju ispod baselinea 37.3%. Odleđivanje encodera s lr 1e-4 je uništilo model; QLoRA recept treba reviziju (lr, warmup, loss-maskiranje image tokena?) — sljedeći trening tek po dogovoru.
- Pažnja: v2 `out/adapter-smo2b-v1` je PUNI model save (4.7 GB `model.safetensors`), ne PEFT — eval zato ide kao `--base`, skripta `slurm/bh-vision-eval-v2.sbatch` (uploadana u job-dir).
- Usputni fix: `training/eval.py` u v2 job-diru bio je star (bez `--out`) — uploadana svježa verzija, prvi eval job (23840543) pao zbog toga.

## LUMI v3 (2026-11-18) — predan, čeka red
- Recept nakon v1 12.0% / v2 0.0%: `configs/smo2b-v3-qlora.yaml` — frozen encoder, lr 2e-5, r16/a32, warmup 0.1, weight decay 0.01, 1 epoha, `save_steps: 30` (eval mid-checkpointa + finalnog).
- `train.py`: +`warmup_ratio`/`weight_decay`/`save_steps` iz konfiga (defaulti čuvaju staro ponašanje).
- Job-dir `smo2b-v3` na scratchu (kopija v2 + svježi fajlovi).
- Pokušaj 1 (job 23840815): FAILED — `SFTConfig got unexpected kwarg 'warmup_ratio'` (LUMI trl/transformers miks; kontejnerski trl 1.16.0 sjeni PYLIBS). Fix: izbačeni warmup/wd knobovi, ostaje samo lr 2e-5 + r16.
- Pokušaj 2 (job 23841124): **TRAIN_DONE** (~1 h), pravi PEFT adapter (108 MB) + checkpointi 30/60/62.
- Eval (job 23844783, cijeli v03-val, 217): **final 95/217 = 43.8%, ckpt-30 100/217 = 46.1%, ckpt-60 95/217 = 43.8%** — SVE iznad baselinea 37.3%. Prvi mali naš model koji ne degradira. Najbolji: **ckpt-30 (46.1%, +8.8pp)** — rani stopping se isplatio.
- Sljedeće: skidanje ckpt-30 lokalno, verifikacija, prijedlog objave adaptera (težine samo uz odobrenje).
- Cilj: prvi mali naš model koji NE degradira ispod baselinea 37.3%.

## LUMI v4 (2026-11-18) — u pripremi: chart-težak trening
- Analiza grešaka v3 (ckpt-30): chart 24.5% (sve checkpointe identično — adapter ništa ne uči za chartove), doc 62.6%.
- Obrasci: vrijednosti ±1–5 (35→40), brkanje mjeseci (august↔Juli), njemački oblici (Oktober), izvedena računica puca.
- Odgovor: `data/synth4` (SEED 104, 200 doc + 800 chart, gušća chart QA: min-mjesec, najtopliji, dupla vrijednost) → merged `synth34-train.jsonl` (4575 uzoraka, 0 preklapanja sa synth3-train).
- Config `smo2b-v4-qlora.yaml` (isti recept kao v3); eval ostaje v03-val radi uporedivosti.
- Sljedeće: stage na LUMI, trening (~2.5 h za 4575 uzoraka), eval final + ckpt.
- Job **23848800** predan (small-g, limit podignut na 4 h zbog 143 koraka). Nadzor u pozadini.

## Mini 500M (2026-11-17) — negativan rezultat
- Kernel `admiragic/bh-vision-mini-500m-v1`, T4, 24 koraka QLoRA r16 + eval 10 val uzoraka istim kernelom.
- Eval: **0/10 normalized** (eho prompta, halucinacije: `august→"Jednak je mjesec..."`, `10→"Špatičnost..."`).
- Netrenirani 2.2B baseline na istim tipovima: 7/10. Zaključak: 500M-video je preslab i/ili 24 koraka premalo — nije presuda za 2.2B put.
- Sljedeći GPU: Kaggle kvota se obnavlja 21.11. ili LUMI (treba EFP cert).

## Verzije (radeće, pinovano)
`transformers==5.22.2`, `trl==1.18.0`, `peft==0.21.2`. Novije od 5.22 lome SmolVLM2 merge (reshape `[-1,14,2048]`).
