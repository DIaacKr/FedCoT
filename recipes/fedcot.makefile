model_name := Meta-Llama-3-8B-Instruct
PYTHON ?= python
datasets := "bioasq,medmcqa,medqa,mmlu,pubmedqa"
dataset_versions := ""
eval_dataset_versions := ""
reward_model := allenai/longformer-base-4096
nli_model := cross-encoder/nli-deberta-v3-base
overwrite_prepared_data := false

batch_size := 16
learning_rate := 2e-4
max_length := 4096

lora_r := 8
lora_dropout := 0.1
num_clients := 5
num_rounds := 3
client_epochs := 1
data_split_method := non_iid_dataset

client_lora_r := "4,32,32,16,4"
eval_batch_size := 1024
eval_max_length := 4096
save_all_rounds := true
evaluate_all_rounds := true
save_only_final := false

random_seed := 42

.PHONY: all prepare_data full_pipeline
all: full_pipeline

prepare_data:
	$(PYTHON) src/data_preparation.py \
		--model_output_dir "outputs/best_of_n/${model_name}" \
		--datasets ${datasets} \
		$(if ${dataset_versions},--dataset_versions ${dataset_versions},) \
		--nli_model "${nli_model}" \
		--random_seed ${random_seed} $(if $(filter true,${overwrite_prepared_data}),--overwrite,)

full_pipeline:
	$(PYTHON) src/fed_pipeline.py \
		--model_name ${model_name} \
		--datasets ${datasets} \
		$(if ${dataset_versions},--dataset_versions ${dataset_versions},) \
		$(if ${eval_dataset_versions},--eval_dataset_versions ${eval_dataset_versions},) \
		--reward_model ${reward_model} \
		--nli_model "${nli_model}" \
		$(if $(filter true,${overwrite_prepared_data}),--overwrite_prepared_data,) \
		--batch_size ${batch_size} \
		--learning_rate ${learning_rate} \
		--max_length ${max_length} \
		--lora_r ${lora_r} \
		--lora_dropout ${lora_dropout} \
		--num_clients ${num_clients} \
		--num_rounds ${num_rounds} \
		--client_epochs ${client_epochs} \
		--data_split_method ${data_split_method} \
		$(if ${client_lora_r},--client_lora_r ${client_lora_r},) \
		--eval_batch_size ${eval_batch_size} \
		--eval_max_length ${eval_max_length} \
		$(if $(filter true,${save_all_rounds}),--save_all_rounds,) \
		$(if $(filter true,${evaluate_all_rounds}),--evaluate_all_rounds,) \
		$(if $(filter true,${save_only_final}),--save_only_final,--no_save_only_final) \
		--random_seed ${random_seed}
