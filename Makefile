.PHONY: install install-train doctor qc-license validate-data

install:
	pip install -e ".[dev]"

install-train:
	pip install -e ".[train,eval]"

doctor:
	python scripts/doctor.py

qc-license:
	python scripts/qc_license.py --manifest data/manifest.yaml

validate-data:
	python scripts/validate_data.py --input data/train.jsonl

train:
	python training/train.py --config $(CONFIG) --train $(TRAIN)

eval:
	python training/eval.py --adapter $(ADAPTER) --base $(BASE) --eval $(EVAL)
