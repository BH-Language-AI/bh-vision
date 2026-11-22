"""PlotQA -> bosanski chart trening set (izvedeno, CC-BY-4.0 + atribucija originalu).
Izvor: nimapourjafar/mm_plotqa (parquet shardovi). Pitanja se prevode regex pravilima,
odgovori ostaju (brojevi), Yes/No -> Da/Ne. Slike se ne mijenjaju.
Samo TRAIN (eval ostaje v03-val radi uporedivosti); held-out se NIKAD ne dira.

Koristenje: N_SHARDS=1 MAX_QA=3000 SEED=11 python3 scripts/make_plotqa_bs.py
"""
import io
import json
import os
import random
import re

OUT = os.getenv("OUT", "data/plotqa-bs")
IMG = f"{OUT}/img"
N_SHARDS = int(os.getenv("N_SHARDS", "1"))
SHARD_START = int(os.getenv("SHARD_START", "0"))
MAX_QA = int(os.getenv("MAX_QA", "3000"))
SEED = int(os.getenv("SEED", "11"))
random.seed(SEED)
os.makedirs(IMG, exist_ok=True)

SUFFIX = re.compile(
    r"\n(Make the answer very short|Provide a succinct answer|Answer with a single word or phrase|"
    r"Your answer should be very brief|Offer a very short reply|Provide a short and direct response|"
    r"Keep it short and to the point|Give a very brief answer|Your answer should be compact|"
    r"Your response must be concise|Offer a terse response|Ensure brevity in your answer)\.?$", re.I)

RULES = [
    # strukturna pitanja
    (re.compile(r"^What is the title of the graph\?$", re.I),
     "Koji je naslov grafa? Odgovori kratko."),
    (re.compile(r"^What is the label or title of the ([XY])-axis\?$", re.I),
     lambda m: f"Koja je oznaka {m.group(1)}-ose? Odgovori kratko."),
    (re.compile(r"^Does the graph contain grids\?$", re.I),
     "Da li graf sadrži mrežu? Odgovori sa Da ili Ne."),
    (re.compile(r"^What is the difference between two consecutive major ticks on the ([XY])-axis\?$", re.I),
     lambda m: f"Kolika je razlika između dvije susjedne glavne podjele na {m.group(1)}-osi? Odgovori samo brojem."),
    (re.compile(r"^How many dotlines are there\?$", re.I),
     "Koliko ima tačkastih linija? Odgovori samo brojem."),
    (re.compile(r"^How many different coloured dotlines are there\?$", re.I),
     "Koliko ima tačkastih linija različitih boja? Odgovori samo brojem."),
    (re.compile(r"^How are the legend labels stacked\?$", re.I),
     "Kako su složene oznake legende? Odgovori kratko."),
    (re.compile(r"^How many years are there in the graph\?$", re.I),
     "Koliko godina ima na grafu? Odgovori samo brojem."),
    (re.compile(r"^Where does the legend appear in the graph\?$", re.I),
     "Gdje se nalazi legenda na grafu? Odgovori kratko."),
    (re.compile(r"^What is the difference between the highest and the lowest (.+?)\?$", re.I),
     lambda m: f"Kolika je razlika između najveće i najmanje {m.group(1)}? Odgovori samo brojem."),
    (re.compile(r"^What is the difference between the highest and the second highest (.+?)\?$", re.I),
     lambda m: f"Kolika je razlika između najveće i druge najveće {m.group(1)}? Odgovori samo brojem."),
    (re.compile(r"^Is the number of dotlines equal to the number of legend labels\?$", re.I),
     "Da li je broj tačkastih linija jednak broju oznaka legende? Odgovori sa Da ili Ne."),
    # broj godina iznad/prag
    (re.compile(r"^In how many years,? is (.+?) greater than ([0-9.\-]+)\?$", re.I),
     lambda m: f"U koliko godina je {m.group(1)} veće od {m.group(2)}? Odgovori samo brojem."),
    (re.compile(r"^In how many years,? is (.+?) less than ([0-9.\-]+)\?$", re.I),
     lambda m: f"U koliko godina je {m.group(1)} manje od {m.group(2)}? Odgovori samo brojem."),
    # razlika dvije vrijednosti
    (re.compile(r"^What is the difference between (.+?) of (.+?) in (\d{4}) and that in (\d{4})\?$", re.I),
     lambda m: f"Kolika je razlika između {m.group(1)} za {m.group(2)} u {m.group(3)}. i vrijednosti iz {m.group(4)}.? Odgovori samo brojem."),
    (re.compile(r"^What is the difference between (.+?) in (.+?) in (\d{4}) and that in (\d{4})\?$", re.I),
     lambda m: f"Kolika je razlika između {m.group(1)} u {m.group(2)} u {m.group(3)}. i vrijednosti iz {m.group(4)}.? Odgovori samo brojem."),
    (re.compile(r"^In the year (\d{4}), what is the difference between (.+?)\?$", re.I),
     lambda m: f"U {m.group(1)}. godini, kolika je razlika između {m.group(2)}? Odgovori samo brojem."),
    # omjer
    (re.compile(r"^What is the ratio of (.+?) of (.+?) in (\d{4}) to that in (\d{4})\?$", re.I),
     lambda m: f"Koliki je omjer {m.group(1)} za {m.group(2)} u {m.group(3)}. i vrijednosti iz {m.group(4)}.? Odgovori samo brojem."),
    (re.compile(r"^What is the ratio of (.+?) in (.+?) in (\d{4}) to that in (\d{4})\?$", re.I),
     lambda m: f"Koliki je omjer {m.group(1)} u {m.group(2)} u {m.group(3)}. i vrijednosti iz {m.group(4)}.? Odgovori samo brojem."),
    # prosjek
    (re.compile(r"^What is the average (.+?) per year\?$", re.I),
     lambda m: f"Koliki je prosječni {m.group(1)} po godini? Odgovori samo brojem."),
    (re.compile(r"^What is the average (.+?) in (\d{4})\?$", re.I),
     lambda m: f"Koliki je prosječni {m.group(1)} u {m.group(2)}. godini? Odgovori samo brojem."),
    # vrijednost entiteta u godini
    (re.compile(r"^What is the (.+?) of (.+?) in (\d{4})\?$", re.I),
     lambda m: f"Koliko iznosi {m.group(1)} za {m.group(2)} u {m.group(3)}. godini? Odgovori samo brojem."),
    (re.compile(r"^What is the (.+?) in (\d{4})\?$", re.I),
     lambda m: f"Koliko iznosi {m.group(1)} u {m.group(2)}. godini? Odgovori samo brojem."),
    # Da/Ne opsta
    (re.compile(r"^(Does|Do|Is|Are|Was|Were|Has|Have|Can) (.+?)\?$", re.I),
     lambda m: f"Da li {m.group(2)}? Odgovori sa Da ili Ne."),
    # brojčana opsta
    (re.compile(r"^How many (.+?)\?$", re.I),
     lambda m: f"Koliko {m.group(1)}? Odgovori samo brojem."),
]

ANS_MAP = {"yes": "Da", "no": "Ne", "horizontally": "vodoravno", "vertically": "uspravno",
           "bottom": "dno", "top": "vrh", "left": "lijevo", "right": "desno"}


def bs_question(q):
    q = SUFFIX.sub("", q).strip()
    for rx, out in RULES:
        m = rx.match(q)
        if m:
            return out(m) if callable(out) else out
    return None


def bs_answer(a):
    a = a.strip().rstrip(".")
    return ANS_MAP.get(a.lower(), a)


def main():
    from PIL import Image
    from huggingface_hub import HfApi
    tok = open("/home/sha/.cache/huggingface/token").read().strip()
    api = HfApi(token=tok)
    info = api.dataset_info("nimapourjafar/mm_plotqa")
    shards = sorted(s.rfilename for s in (info.siblings or [])
                    if s.rfilename.startswith("data/train-"))[SHARD_START:SHARD_START + N_SHARDS]
    import pandas as pd
    rows_out, n_img, n_skip = [], 0, 0
    for sh in shards:
        p = api.hf_hub_download("nimapourjafar/mm_plotqa", sh, repo_type="dataset")
        df = pd.read_parquet(p)
        for ri, r in df.iterrows():
            msgs = r["data"]
            qa = [(msgs[j]["data"], msgs[j + 1]["data"]) for j in range(len(msgs) - 1)
                  if msgs[j].get("role") == "user" and msgs[j].get("modality") == "text"
                  and msgs[j + 1].get("role") == "assistant"]
            random.shuffle(qa)
            taken = 0
            for q, a in qa:
                if len(rows_out) >= MAX_QA:
                    break
                bq = bs_question(q)
                if not bq:
                    n_skip += 1
                    continue
                img_p = f"{IMG}/pq{n_img:05d}.jpg"
                im = Image.open(io.BytesIO(r["images"][0]["bytes"])).convert("RGB")
                im.save(img_p, quality=80)
                rows_out.append({"messages": [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": bq}]},
                    {"role": "assistant", "content": [{"type": "text", "text": bs_answer(a)}]}],
                    "images": [img_p]})
                taken += 1
                if taken >= 2:
                    break
            n_img += 1
            if len(rows_out) >= MAX_QA:
                break
        if len(rows_out) >= MAX_QA:
            break
    with open(f"{OUT}/train.jsonl", "w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"gotovo: uzoraka={len(rows_out)} slika={n_img} preskoceno-bez-prijevoda={n_skip}")


if __name__ == "__main__":
    main()
