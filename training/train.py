"""QLoRA trening vision-only adaptera (SmolVLM2 start). Trosi GPU — pokretati samo uz odobrenje.
shortcut: netestirano na GPU (ovaj server nema GPU); prije prvog treninga suhi `make doctor` + probni run na L4.
Koristenje: python training/train.py --config configs/smo2b-qlora.yaml --train data/synth/train.jsonl
"""
import argparse, yaml
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import torch
import torch.nn.functional as F


def ce_loss(outputs, labels, num_items_in_batch):
    # Obican CE umjesto TRL fused kernela (fused trazi sm>=8.0 i puca i na ROCm-u).
    logits = outputs.logits.float()
    sl = logits[..., :-1, :].contiguous()
    lb = labels[..., 1:].contiguous()
    return F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1), ignore_index=-100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="releases/adapter-v01")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "none"],
                    help="4bit = bitsandbytes (CUDA); none = bf16 LoRA (LUMI ROCm / CPU)")
    a = ap.parse_args()
    c = yaml.safe_load(open(a.config))

    if a.quant == "4bit":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForImageTextToText.from_pretrained(
            c["base_model"], quantization_config=bnb, device_map={"": 0} if torch.cuda.is_available() else "auto")
    else:
        dt = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(c["base_model"], dtype=dt)
    proc = AutoProcessor.from_pretrained(c["base_model"])
    if c.get("freeze_vision_encoder"):
        for attr in ("vision_model", "visual", "vision_tower"):
            if hasattr(model, attr):
                for p in getattr(model, attr).parameters():
                    p.requires_grad = False

    ds = load_dataset("json", data_files=a.train, split="train")

    def fmt(ex):
        text = proc.apply_chat_template(ex["messages"], tokenize=False)
        return {"text": text, "images": ex["images"]}

    ds = ds.map(fmt, remove_columns=[k for k in ds.column_names if k not in ("images",)])
    lora = LoraConfig(r=c["lora"]["r"], lora_alpha=c["lora"]["alpha"],
                      lora_dropout=c["lora"].get("dropout", 0.05),
                      target_modules="all-linear", task_type="CAUSAL_LM")
    # Napomena: BEZ warmup_ratio/weight_decay kwargova — LUMI PYLIBS/kontejner miks
    # (trl 1.16.0/1.18.0 vistrucno) baca TypeError na nepoznat kwarg; lr je ionako 10x dole.
    args = SFTConfig(output_dir=a.out, num_train_epochs=c.get("epochs", 2),
                     max_steps=c.get("max_steps", -1),
                     learning_rate=c.get("lr", 1.5e-4),
                     per_device_train_batch_size=1, gradient_accumulation_steps=32,
                     logging_steps=10, save_steps=c.get("save_steps", 60), save_total_limit=3,
                     max_length=None, dataset_text_field="text", remove_unused_columns=False,
                     use_cpu=not torch.cuda.is_available())
    trainer = SFTTrainer(model=model, train_dataset=ds, args=args, peft_config=lora,
                         processing_class=proc, compute_loss_func=ce_loss)
    trainer.train()
    trainer.model.save_pretrained(a.out)  # PeftModel -> adapter, ne puna tezina
    proc.save_pretrained(a.out)
    print("adapter sacuvan u", a.out)


if __name__ == "__main__":
    main()
