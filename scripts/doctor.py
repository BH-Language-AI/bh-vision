"""Suha provjera okruzenja — bez GPU treninga, samo cita confige i prijavljuje sta fali."""
import glob, sys
try:
    import yaml
except ImportError:
    sys.exit("doctor: treba pyyaml (make install)")

ok = True
for f in sorted(glob.glob("configs/*.yaml")):
    c = yaml.safe_load(open(f))
    need = ["base_model", "method", "held_out"]
    missing = [k for k in need if k not in c]
    print(("OK  " if not missing else "FAIL"), f, ("| fali: " + ",".join(missing)) if missing else "")
    ok = ok and not missing
try:
    import transformers, trl, peft  # noqa
    print("OK   train libs (transformers/trl/peft prisutni)")
except ImportError as e:
    print("WARN train libs fale (ocekivano bez make install-train):", e)
sys.exit(0 if ok else 1)
