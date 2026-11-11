# BH-Vision — Arhitektura (Faza 1)

Slika/dokument/video -> Vision Encoder -> Konektor -> Jezicki backbone -> strukturiran odgovor / tool call.

## Odluka: Arch A
Adaptacija postojeceg VLM-a (SmolVLM2-2.2B prvi kandidat) za bs/hr/sr.
Arch B (konektor na BH-LLM-Basic) odlozen — backbone ne postoji.

## Velicine
2B (SmolVLM2-2.2B) + 8B (Idefics3-8B-Llama3) + 12B (Pixtral) opcionalno.
Fiksne 4B/7B odbijene — nema EU 4B checkpointa, 7B klasa je 8B/12B.

## Framework
AgentMujo Training je tekst-only -> odvojeni vision adapter/modul,
dijeljena samo infrastruktura (CLI, manifesti, eval okvir) gdje ima smisla.
