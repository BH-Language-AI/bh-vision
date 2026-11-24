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

## BH held-out v03 (2026-11-23) — 101 QA, prvi skor: 48.5%
- +14 FZS oktobar (gušća 7-kolonska tabela). Qwen v2+sys: **49/101 = 48.5%** (starih 87: 52.9%, novih 14: 21.4%).
- Novi promašaji su pravi (susjedne kolone/redovi u gušćoj tabeli), ne format. Oktobarska tabela je najteži layout dosad.
- Metrička napomena: v01 gold koristi zareze (kao na slici), FZS gold tačke (kao text-layer) — stroga metrika ih tretira različito; blaga metrika u analizama.

## BH held-out v02 (2026-11-22) — proširen FZS septembrom
- 87 QA (v01 73 + 14 FZS 09/2026, drugi layout). Qwen+adapter+instrukcija: **46/87 = 52.9%** (v01 dio ~54.8%, novih 14 teži — očekivano, novi layout).
- v01 ostaje zamrznuta za uporedivost; v02 je novi glavni metar.
- v01 historijat: 73 ručno verificirana QA (bhas 09/2026); baseline 21.9%, v3 23.3%, Qwen 49.3% — synth-dobit se nije prenosila.

## LUMI v7 (2026-11-19) — završen, ODBIJEN: pravi dokumenti pomažu minimalno
- Trening `smo2b-v7` (synth7: 3015 = synth5 + bhas-train 113): `TRAIN_DONE`.
- Eval v03-val: final 67/217 = 30.9%, ckpt-60 72/217 = 33.2% (doc 40.7%, chart 23.4% zid).
- Eval held-out v01: **final 16/73 = 21.9%, ckpt-60 18/73 = 24.7%** vs v3 23.3% vs baseline 21.9% — unutar šuma (±5pp na 73 uzorka), bez stvarnog pomaka.
- Zaključak: 113 real-doc QA nije dovoljno da se generalizacija pomakne; treba RED VELIČINE više pravih dokumenata (svi bhas mjeseci + drugi izvori), ili drugačiji pristup. Trening-spirala pauzirana: v3 ostaje jedini objavljeni model.

## LUMI Pixtral-12B v1 (2026-11-21/22) — završen, ODBIJEN: veći model, slabiji rezultat
- Trening (synth7, max 40 koraka): `TRAIN_DONE` (trebalo trust_remote_code + community konverzija jer službeni repo nije transformers-format).
- Eval v03-val: baza **147/217 = 67.7%** → adapter **152/217 = 70.0%** (+2.3pp).
- Eval held-out: baza **22/73 = 30.1%** → adapter **21/73 = 28.8%** (-1.3pp, šum).
- Zaključak: 12B je točno između 2B (46.1%/23.3%) i 4B (90.3%/49.3%) — veće NE znači bolje na ovom zadatku; Qwenova arhitektura/trening odnosi prevagu. 12B se ne objavljuje; 4B ostaje šampion.

## LUMI Qwen7B v1 (2026-11-24) — produkcijska 8B: OBJAVLJENA
- Trening (synth7, max 40 koraka): `TRAIN_DONE`. Eval v03-val: baza **193/217 = 88.9%** → adapter 192/217 = 88.5% → adapter+sys **203/217 = 93.5%**.
- Eval held-out v02: baza **52/87 = 59.8%** → adapter+sys **51/87 = 58.6%** (najbolji held-out svih klasa).
- Objavljena kao **BH-Language-AI/bh-vision-1-8b** (Qwen7B adapter umjesto Idefics placeholdera) + kartica, u kolekciji.

## LUMI v8 (2026-11-19/20) — završen, ODBIJEN: najduži trening, najdublja memorizacija
- Trening `smo2b-v8` (synth8: 5427, 170 koraka): `TRAIN_DONE`.
- Eval v03-val: **final 12/217 = 5.5%, ckpt-150 7/217 = 3.2%**. Eval held-out: **0/73**.
- Zakon cijele serije: v3-30st 46% > v3-62st 44% > v5/v7 33-35% (90+ st) > v1 12% (124 st) > v4/v8 ~4-5% (143-170 st). Broj koraka ubija, ne podaci.
- v3 ckpt-30 (30 koraka, 46.1%) ostaje šampion.
- Podaci v8: `bhas-bulk` 1825 QA (56 stranica) + `synth8` 5427 mix; miner verificiran.

## LUMI Qwen3B v12 (2026-11-23) — završen, ODBIJEN: format se ne uči treningom
- Trening sa sistemskom instrukcijom u treningu (synth7, max 40 koraka): `TRAIN_DONE`.
- Eval s istom instrukcijom: val **201/217 = 92.6%** (v1+sys 92.2%), held-out v02 **46/87 = 52.9%** (v1+sys 52.9%) — identično, nikakav dobitak.
- Zaključak: disciplina formata se ne prenosi treningom u ovom setupu; inference instrukcija je sve što ima. v1 ostaje objavljeni 4B.

## BH-Vision-1-4B-v2 (2026-11-23) — zadatak: kontrolisano unapređenje, baza se NE mijenja
- Revizija stanja: config v1 (r16/a32, lr 2e-5, max 40, synth7 3015), adapter verificiran (828 LoRA tenzora, r16). Reference v1: val 90.3% (92.2% sys), heldout v01 49.3% (54.8% sys), v02 52.9% (sys).
- Revizija podataka: 2 duplikata + **5 val slika u treningu (isti hash, različita imena!)** — tvrdnja "split bez curenja" nije važila za tih 5. Held-out čist (0 preklapanja). Očišćeno: `synth7fix` (3005).
- Eksperimenti (jedna promjena odjednom): **v2a** = isti recept + očišćeni podaci (val 89.4%/91.2% sys, heldout 54.8%/52.9% sys — decimalno identično v1); **v2b** = r32/a64 (val 87.1%/89.4% sys, heldout 54.8%/50.6% sys — LOŠIJE od r16 svugdje).
- Presuda: ni čišćenje ni rank 32 ne daju mjerljivo poboljšanje na stvarnom skupu (heldout v01/v02 identičan ili lošiji). **v2 se NE objavljuje** — v1 ostaje. Nalaz: rank 16 > rank 32 za ovaj zadatak; curenje od 5 slika nije uticalo na zaključke (v2a ≈ v1).
- **v2c** (lr 1e-5, jedina promjena): val 90.3%/93.5% sys, heldout 56.2%/54.0% sys — konzistentno +1pp svugdje iznad v1.
- **OBJAVLJEN kao BH-Language-AI/bh-vision-1-4b-v2** (odobrenje vlasnika 2026-11-23): adapter + kartica s poštenom tabelom v1/v2, uvršten u kolekciju.

## LUMI v9 (2026-11-20) — završen: PLAFON POTVRĐEN, v3 ostaje
- Trening `smo2b-v9` (synth7: 3015, max 40 koraka): `TRAIN_DONE`.
- Eval v03-val: **final 101/217 = 46.5%** (v3: 46.1%). Eval held-out: **17/73 = 23.3%** (v3: 23.3%).
- Zaključak: kratki trening na boljim podacima reproducira ali NE probija plafon (~46% val / ~23% heldout). QLoRA r16 na 2.2B je iscrpljen — za dalje treba veći model (8B), veća rezolucija ili full-finetune.
- Nijansa: v9 doc 63.4% (najbolji doc dosad, v3 62.6%), chart zid 24.5% svugdje. Objavljen ostaje v3 (verificiran prvi); v9 je validirana alternativa u evidenciji.

## LUMI v10 (2026-11-20) — završen, ODBIJEN: plafon treći put potvrđen
- Trening `smo2b-v10` (synth10: 5535, max 70 koraka): `TRAIN_DONE`.
- Eval v03-val: **final 88/217 = 40.6%**. Eval held-out: **17/73 = 23.3%** — identično v3 i v9.
- Zaključak: ni najveći miks ne pomiče plafon (~46% val / ~23% held-out). Svi pokušaji v4–v10 odbijeni; v3 ckpt-30 ostaje jedini objavljeni model.
- Cert obnovljen 20.11 → stage OK (sve slike verificirane: 0 nedostaje), trening **job 23881149** predan (small-g, limit 3 h). Nadzor u pozadini.

## LUMI Idefics3-8B v1 (2026-11-21) — završen, ODBIJEN: slabija baza + trening šteti
- Trening (synth7, max 40 koraka): `TRAIN_DONE`. Eval v03-val: **adapter 79/217 = 36.4%** (baza 90/217 = 41.5% → trening ŠTETI -5.1pp, ista bolest kao v1).
- Eval held-out: **adapter 17/73 = 23.3%, baza 18/73 = 24.7%** (= v3 nivo, trening blago šteti i ovdje).
- Zaključak: Idefics3-8B baza (41.5%/24.7%) je daleko ispod Qwena (87.6%/49.3%), a adaptacija je degradira. 8B se ne objavljuje; Qwen 4B ostaje šampion.
- Baza+instrukcija na valu: strogo 73/217 = 33.6% (instrukcija ŠTETI Ideficsu, suprotno od Qwena), blago 121/217 = 55.8% (nova lenient metrika radi).

## LUMI Qwen3B v1 (2026-11-20) — 4B pilot: OBJAVLJEN
- Trening (synth7, max 40 koraka): `TRAIN_DONE`. Eval v03-val: **final 196/217 = 90.3%** (chart 87.2%, doc 92.7%) — chart zid 24.5% srušen samom bazom.
- Eval held-out: **36/73 = 49.3%** (v3: 23.3%, baseline SmolVLM2: 21.9%) — prava generalizacija, duplo bolje.
- Pravi Qwen baseline: val 190/217 = 87.6% → adapter +2.7pp; heldout 36/73 = 49.3% = identično adapteru (±0). Baza nosi, adapter dodaje bez lomljenja.
- Sistemska instrukcija ("odgovaraj isključivo kratko..."): val **200/217 = 92.2%** (+1.9pp), held-out **40/73 = 54.8%** (+5.5pp) — najjeftiniji dobitak, bez treninga; preporučeni način korištenja na kartici.
- Analiza grešaka held-out: većina preostalih su FORMAT (zarez/tačka, jedinice, pune rečenice), ne nerazumijevanje; blaga metrika daje 74%.
- Objavljen kao **BH-Language-AI/bh-vision-1-4b** (adapter 165 MB + kartica s poštenim omjerom baza/adapter), dodat u HF kolekciju.

## LUMI 8B v2 (2026-11-22) — završen, ODBIJEN: ni prošireni podaci ne pomažu
- Trening (synth11: 7143, max 40 koraka): `TRAIN_DONE`. Eval v03-val: **80/217 = 36.9%** (= v1 36.4%). Eval held-out: **16/73 = 21.9%** (v1 23.3%, = baseline SmolVLM2).
- Usputni fix: eval padao jer je globalni trust_remote_code učitao drugu arhitekturu (4608 vs 10368) — sada opcija po modelu.
- Konačna presuda 8B: baza slaba + trening ne pomaže ni s najboljim podacima. 8B zatvoren; Qwen 4B ostaje jedini šampion (90.3%/49.3%, sa sistemskom instrukcijom 92.2%/54.8%).

## Mini 500M (2026-11-17) — negativan rezultat
- Kernel `admiragic/bh-vision-mini-500m-v1`, T4, 24 koraka QLoRA r16 + eval 10 val uzoraka istim kernelom.
- Eval: **0/10 normalized** (eho prompta, halucinacije: `august→"Jednak je mjesec..."`, `10→"Špatičnost..."`).
- Netrenirani 2.2B baseline na istim tipovima: 7/10. Zaključak: 500M-video je preslab i/ili 24 koraka premalo — nije presuda za 2.2B put.
- Sljedeći GPU: Kaggle kvota se obnavlja 21.11. ili LUMI (treba EFP cert).

## Verzije (radeće, pinovano)
`transformers==5.22.2`, `trl==1.18.0`, `peft==0.21.2`. Novije od 5.22 lome SmolVLM2 merge (reshape `[-1,14,2048]`).
