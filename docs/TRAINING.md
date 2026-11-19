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

## LUMI v4 (2026-11-18/19) — završen, NEGATIVAN: chart-imbalance kolaps
- Trening `smo2b-v4` (synth34: 4575 uzoraka, 89% chart QA): `TRAIN_DONE`, loss 649→6.4 (memorizacija, bez divergencije).
- Eval: **final 8/217 = 3.7%, ckpt-30 12/217 = 5.5%, ckpt-60 6/217 = 2.8%** — doc 0/123 na svim checkpointima.
- Mehanizam (iz predikcija): model sve slike vidi kao chart (`The x-axis represents...`, `The graph is called...`) — chart-težak miks prebacio vizualni prior u chart-mode, OCR sposobnost ugašena.
- Lekcija: balans tipova je obavezan; v3 (50/50-ish) ostaje najbolji (46.1%).

## LUMI v5 (2026-11-19) — završen, ISPOD v3: balans vratio dio
- Trening `smo2b-v5` (synth5: 2902, 50/50): `TRAIN_DONE`. Eval (job 23852638): **final 70/217 = 32.3%, ckpt-30 75/217 = 34.6%, ckpt-60 32.3%**.
- Raspad ckpt-30: chart 23/94 = 24.5% (identično v3 i baselineu — chart se ne miče), doc 52/123 = 42.3% (oporavak od v4 0%, ali ispod v3 62.6%).
- Chart analiza: direktne vrijednosti 5/41 = 12.2% (gore od max/min 18/53 = 34%) — model ne čita tačne visine stubaca (±1–5). Zaključak: precizno očitanje vrijednosti se ne uči ovim podacima; treba prava chart raznolikost (PlotQA-bs) ili supervizija s označenim vrijednostima.
- v3 ckpt-30 (46.1%) ostaje najbolji objavljeni model.

## LUMI v6 (2026-11-19) — završen, ODBIJEN: chart zid potvrđen
- Trening `smo2b-v6` (synth6: 3602, chart 59.7%): `TRAIN_DONE`. Eval: **final 36/217 = 16.6%, ckpt-30 73/217 = 33.6%, ckpt-60 39/217 = 18.0%**.
- Raspad: chart 21–23/94 u SVIM checkpointima (isto kao baseline/v3/v5 — PlotQA-bs raznolikost NIJE pomakla chart), doc 41.5%→13% (klasična memorizacija krivulja, strmija nego v3).
- Zaključak: chart 24.5% je zid za QLoRA r16 na 2.2B s ovim podacima — precizno očitanje vrijednosti se ne uči ovako. Treba drugačiji pristup (veća rezolucija, full-finetune, ili baza s jačim chartovima), ne još jedan isti trening.
- v3 ckpt-30 (46.1%) ostaje šampion i objavljeni model. Trening-spirala se ovdje zaustavlja; sljedeće: held-out pravim dokumentima, 8B klasa, ili chart-specific pristup.

## BH held-out v01 (2026-11-19) — NAPUNJEN + PRVI SKOROVI
- `data/heldout-v01/eval.jsonl`: **73 ručno verificirana QA** s 19 stranica bhas.gov.ba saopćenja (CPI/PPI/prosječne cijene 09/2026) — prave tabele, zarez-decimale, dijakritici, 1 pravi graf.
- Objavljeno na `BH-Language-AI/bs-vision-ocr-heldout-v01` (CC-BY-4.0 + atribucija). NIKAD u train.
- Sljedeće: eval v3-šampiona (ckpt-30) + baselinea na held-outu — pravi generalizacijski skor.
- **Rezultat: baseline 16/73 = 21.9%, v3-šampion 17/73 = 23.3%** — praktično izjednačeno, synth-dobit se NE prenosi na prave dokumente.
- Zaključak: v7 mora učiti na held-out-SLIČNIM podacima (drugi mjeseci bhas saopćenja = isti layouti, drugi brojevi — legitiman train, held-out ostaje netaknut).

## LUMI v7 (2026-11-19) — završen, ODBIJEN: pravi dokumenti pomažu minimalno
- Trening `smo2b-v7` (synth7: 3015 = synth5 + bhas-train 113): `TRAIN_DONE`.
- Eval v03-val: final 67/217 = 30.9%, ckpt-60 72/217 = 33.2% (doc 40.7%, chart 23.4% zid).
- Eval held-out v01: **final 16/73 = 21.9%, ckpt-60 18/73 = 24.7%** vs v3 23.3% vs baseline 21.9% — unutar šuma (±5pp na 73 uzorka), bez stvarnog pomaka.
- Zaključak: 113 real-doc QA nije dovoljno da se generalizacija pomakne; treba RED VELIČINE više pravih dokumenata (svi bhas mjeseci + drugi izvori), ili drugačiji pristup. Trening-spirala pauzirana: v3 ostaje jedini objavljeni model.

## LUMI v8 (2026-11-19/20) — završen, ODBIJEN: najduži trening, najdublja memorizacija
- Trening `smo2b-v8` (synth8: 5427, 170 koraka): `TRAIN_DONE`.
- Eval v03-val: **final 12/217 = 5.5%, ckpt-150 7/217 = 3.2%**. Eval held-out: **0/73**.
- Zakon cijele serije: v3-30st 46% > v3-62st 44% > v5/v7 33-35% (90+ st) > v1 12% (124 st) > v4/v8 ~4-5% (143-170 st). Broj koraka ubija, ne podaci.
- v3 ckpt-30 (30 koraka, 46.1%) ostaje šampion.
- Podaci v8: `bhas-bulk` 1825 QA (56 stranica) + `synth8` 5427 mix; miner verificiran.

## LUMI v9 (2026-11-20) — predan: kratki trening po lekciji serije
- Recept: synth7 (3015) + max 40 koraka + save 20 (umjesto punih epoha).
- Cilj: nadmašiti v3 kraćim treningom na boljim podacima.

## Mini 500M (2026-11-17) — negativan rezultat
- Kernel `admiragic/bh-vision-mini-500m-v1`, T4, 24 koraka QLoRA r16 + eval 10 val uzoraka istim kernelom.
- Eval: **0/10 normalized** (eho prompta, halucinacije: `august→"Jednak je mjesec..."`, `10→"Špatičnost..."`).
- Netrenirani 2.2B baseline na istim tipovima: 7/10. Zaključak: 500M-video je preslab i/ili 24 koraka premalo — nije presuda za 2.2B put.
- Sljedeći GPU: Kaggle kvota se obnavlja 21.11. ili LUMI (treba EFP cert).

## Verzije (radeće, pinovano)
`transformers==5.22.2`, `trl==1.18.0`, `peft==0.21.2`. Novije od 5.22 lome SmolVLM2 merge (reshape `[-1,14,2048]`).
