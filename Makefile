PY ?= python3

.PHONY: test train experiments figures web serve all

test:            ## framework vs PyTorch, models, tokenizer, browser engine
	$(PY) -m pytest

train:           ## the two demo models (~3 min + ~50 min on a laptop CPU)
	$(PY) examples/train_mnist.py
	$(PY) examples/train_gpt.py

experiments:     ## ablations, scaling, tokenizers, quantisation, benchmark
	cd experiments && $(PY) ablations.py
	cd experiments && $(PY) language_models.py
	cd experiments && $(PY) quantization.py
	cd experiments && $(PY) benchmark.py

figures:         ## README figures from saved results
	$(PY) experiments/make_figures.py

web:             ## int8 bundles + demo data for the browser demo
	$(PY) web/export.py

serve:           ## open http://localhost:8000
	cd web && $(PY) -m http.server 8000

all: test train experiments figures web
