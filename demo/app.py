"""BH-Vision demo: postavi sliku dokumenta/chartova, pitaj na bosanskom."""
import gradio as gr
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER = "BH-Language-AI/bh-vision-1-4b-v2"
SYSTEM = ("Odgovaraj iskljucivo kratko: samo trazeni broj, naziv ili Da/Ne. "
          "Bez objasnjenja i bez dodanih rijeci.")

print("ucitavam model...", flush=True)
proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    BASE, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print("spremno.", flush=True)


def odgovori(slika, pitanje):
    if slika is None or not (pitanje or "").strip():
        return "Postavi sliku i napiši pitanje."
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pitanje}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=text, images=[slika.convert("RGB")], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=64)
    return proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


demo = gr.Interface(
    fn=odgovori,
    inputs=[gr.Image(type="pil", label="Slika (dokument, tabela, chart)"),
            gr.Textbox(label="Pitanje na bosanskom", value="Koliki je ukupan iznos na računu? Odgovori samo iznosom.")],
    outputs=gr.Textbox(label="Odgovor"),
    title="BH-Vision-1-4B-v2 — bosanski vizuelni AI",
    description="Val 93.5% · held-out 52.9% (pravi bhas dokumenti). Adapter: BH-Language-AI/bh-vision-1-4b-v2.")
demo.launch()
