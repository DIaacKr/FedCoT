PYTHON ?= python
CONFIG ?= configs/best_of_n.yaml
MODEL_PATH ?=
DATASETS ?= bioasq,medmcqa,medqa,mmlu,pubmedqa
NUM_CANDIDATES ?= 8
MODES ?= train,test
SEED ?= 42
MAX_NEW_TOKENS ?= 512
DP ?= 1
TP ?= 1
LIMIT ?= -1
PORT ?= 29035
OVERWRITE ?= false

.PHONY: all generate
all: generate

generate:
	@if [ -z "$(MODEL_PATH)" ]; then echo "Set MODEL_PATH to a local instruction model directory." >&2; exit 2; fi
	PYTHON="$(PYTHON)" bash recipes/run_best_of_n.sh \
		--config "$(CONFIG)" \
		--datasets "$(DATASETS)" \
		--num_candidates "$(NUM_CANDIDATES)" \
		--modes "$(MODES)" \
		--seed "$(SEED)" \
		--max_new_tokens "$(MAX_NEW_TOKENS)" \
		--dp "$(DP)" \
		--tp "$(TP)" \
		--limit "$(LIMIT)" \
		--port "$(PORT)" \
		--model_path "$(MODEL_PATH)" $(if $(filter true,$(OVERWRITE)),--overwrite,)

