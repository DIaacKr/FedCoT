# FedCoT

FedCoT generates multiple chain-of-thought candidates, filters training candidates, assigns NLI-based soft rewards, and trains a federated discriminator model to select a candidate.

## Source layout

- `src/`: candidate generation, data loading and preparation, discriminator-model training, FedCoT aggregation, and discriminator-model evaluation.
- `recipes/`: candidate generation and FedCoT commands.
- `configs/`: candidate-generation defaults.

## Run FedCoT

Run commands from the repository root.

### 1. Generate candidates

```bash
make -f recipes/gen_cot.makefile generate \
  MODEL_PATH=/path/to/local/instruction-model \
  DATASETS=bioasq,medmcqa,medqa,mmlu,pubmedqa \
  NUM_CANDIDATES=8
```

Candidate files are written under `outputs/best_of_n/<model-directory-name>/`.

### 2. Prepare data, train, and evaluate

```bash
make -f recipes/fedcot.makefile full_pipeline \
  model_name=Meta-Llama-3-8B-Instruct \
  reward_model=allenai/longformer-base-4096
```

`full_pipeline` prepares the selected training candidates, then trains and evaluates FedCoT. 

To prepare data without training, run:

```bash
make -f recipes/fedcot.makefile prepare_data \
  model_name=Meta-Llama-3-8B-Instruct
```

The latest generation version is used by default. Pass `dataset_versions=...` to select the same explicit versions for preparation and training. 
