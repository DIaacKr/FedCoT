#!/usr/bin/env python3
import sys
import json
import copy
import torch
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

from reward_model import (
    ModelArguments, DataArguments, RewardTrainingArguments,
    RewardModelTrainer, load_dataset_samples, parse_dataset_versions,
    get_latest_version_dir, determine_training_method, setup_model_and_tokenizer, prepare_datasets
)

from peft import AutoPeftModelForSequenceClassification, LoraConfig, get_peft_model, TaskType

@dataclass
class FederatedArguments:
    num_clients: int = field(default=5)
    num_rounds: int = field(default=3)
    client_epochs: int = field(default=1)
    
    data_split_method: str = field(default="iid")
    
    client_lora_r: str = field(default="")

    save_all_rounds: bool = field(default=False)
    evaluate_all_rounds: bool = field(default=False)
    save_only_final: bool = field(default=True)

    random_seed: int = field(default=42)


class FedCoTAggregator:
    
    def __init__(self, client_lora_ranks: List[int]):
        self.client_lora_ranks = client_lora_ranks
        self.total_rank = sum(client_lora_ranks)
        
        logger.info(f"FedCoT aggregation with client ranks {client_lora_ranks} (total rank {self.total_rank})")
    
    def aggregate_models(self, client_models: List[Any], client_weights: List[float]) -> Any:
        fresh_base_model = copy.deepcopy(client_models[0].get_base_model())
        
        original_config = client_models[0].peft_config['default']
        aggregated_config = copy.deepcopy(original_config)
        aggregated_config.r = self.total_rank
        aggregated_config.lora_alpha = self.total_rank
        

        global_model = get_peft_model(fresh_base_model, aggregated_config)

        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        client_state_dicts = [model.state_dict() for model in client_models]
        
        aggregated_state_dict = self._create_stacked_state_dict(client_state_dicts, normalized_weights)
        
        global_model.load_state_dict(aggregated_state_dict, strict=False)
        return global_model 
    
    def _create_stacked_state_dict(self, client_state_dicts: List[Dict], weights: List[float]) -> Dict:
        if not client_state_dicts:
            raise ValueError("No client state dicts provided")
            
        base_dict = client_state_dicts[0]
        aggregated = {}

        for key in base_dict.keys():
            if 'lora_A' in key:
                matrices = []
                for i, (state_dict, weight) in enumerate(zip(client_state_dicts, weights)):
                    if key in state_dict:
                        scaled_matrix = state_dict[key] * weight
                        matrices.append(scaled_matrix)
                
                if matrices:
                    aggregated[key] = torch.cat(matrices, dim=0)
                
            elif 'lora_B' in key:
                matrices = []
                for i, state_dict in enumerate(client_state_dicts):
                    if key in state_dict:
                        matrices.append(state_dict[key])
                
                if matrices:
                    aggregated[key] = torch.cat(matrices, dim=1)
                    
            elif 'classifier' in key:
                aggregated_param = None
                for state_dict, weight in zip(client_state_dicts, weights):
                    param = state_dict[key]
                    weighted_param = param * weight
                    if aggregated_param is None:
                        aggregated_param = weighted_param.clone()
                    else:
                        aggregated_param += weighted_param
                aggregated[key] = aggregated_param
            else:
                aggregated[key] = base_dict[key]
        
        return aggregated

class DataSplitter:
    
    @staticmethod
    def split_data_iid(samples: List[str], labels: List[int], num_clients: int) -> List[Tuple[List[str], List[int]]]:
        logger.info(f"Performing IID split for {len(samples)} samples across {num_clients} clients")
        
        indices = list(range(len(samples)))
        np.random.shuffle(indices)
        
        samples_per_client = len(samples) // num_clients
        client_data = []
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            if i == num_clients - 1:  
                end_idx = len(samples)
            else:
                end_idx = start_idx + samples_per_client
            
            client_indices = indices[start_idx:end_idx]
            client_samples = [samples[idx] for idx in client_indices]
            client_labels = [labels[idx] for idx in client_indices]
            
            client_data.append((client_samples, client_labels))
            
        
        return client_data
    
    @staticmethod
    def split_data_by_dataset(model_output_dir: str, dataset_names: List[str], 
                             dataset_versions: Dict[str, str], num_clients: int) -> List[Tuple[List[str], List[int]]]:
        if num_clients != len(dataset_names):
            raise ValueError("non_iid_dataset splitting requires one client per dataset")
        
        client_data = []
        actual_versions: Dict[str, str] = {}
        template = "Q: {question}\nA: {response}"
        total_samples = 0
        
        for client_id, dataset_name in enumerate(dataset_names):
            
            version = dataset_versions.get(dataset_name, "latest")
            base_dir = Path(model_output_dir) / f"{dataset_name}_best_of_n" / "train"
            
            if version == "latest":
                data_dir = get_latest_version_dir(base_dir)
                if data_dir.name.startswith("version_"):
                    actual_version = data_dir.name
                else:
                    actual_version = f"base_{data_dir.stat().st_mtime:.0f}"
            else:
                data_dir = base_dir / (version if version.startswith("version_") else f"version_{version}")
                actual_version = version

            actual_versions[dataset_name] = actual_version
            data_path = data_dir / "train_best_of_n_filtered_soft.json"
            
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found for {dataset_name}: {data_path}")
            
            
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            samples, labels = [], []
            for sample in data:
                question = sample['question']
                for candidate in sample['candidates']:
                    text = template.format(question=question, response=candidate['response_text'])
                    reward = float(candidate.get('reward', 0.0))
                    samples.append(text)
                    labels.append(reward)
            
            total_samples += len(samples)
            
            client_data.append((samples, labels))
        logger.info(f"Loaded {total_samples} samples across {len(client_data)} clients")
        
        return client_data, actual_versions


class FederatedRewardModelTrainer:
    
    def __init__(self, model_args: ModelArguments, training_args: RewardTrainingArguments, 
                 federated_args: FederatedArguments, aggregator: FedCoTAggregator):
        self.model_args = model_args
        self.training_args = training_args
        self.federated_args = federated_args
        self.aggregator = aggregator
        

        self.random_seed = federated_args.random_seed
        
        self.global_model = None
        self.tokenizer = None
        
        self.client_lora_ranks = self._parse_client_lora_ranks(federated_args)
        
        self.current_round = 0
        
        self.saved_models = []  # [(round_num, model_path), ...]
        

    def _parse_client_lora_ranks(self, federated_args: FederatedArguments) -> List[int]:
        if not federated_args.client_lora_r:
            return [self.model_args.lora_r] * federated_args.num_clients

        ranks = [int(rank.strip()) for rank in federated_args.client_lora_r.split(",")]
        if len(ranks) != federated_args.num_clients:
            raise ValueError(
                f"Expected {federated_args.num_clients} client LoRA ranks, got {len(ranks)}"
            )
        logger.info(f"Using client LoRA ranks {ranks}")
        return ranks
    
    def setup_global_model(self):
        original_lora_r = self.model_args.lora_r
        self.model_args.lora_r = self.client_lora_ranks[0]
        self.global_model, self.tokenizer = setup_model_and_tokenizer(self.model_args)
        self.model_args.lora_r = original_lora_r
    
    def _create_client_model_from_base_fedcot(self, client_id: int) -> Any:
        logger.info(f"Creating fresh client {client_id} model for first round...")
        
        client_lora_r = self.client_lora_ranks[client_id]
        client_lora_alpha = client_lora_r
        
        client_model_args = copy.deepcopy(self.model_args)
        client_model_args.lora_r = client_lora_r
        client_model_args.lora_alpha = client_lora_alpha
        
        client_model, _ = setup_model_and_tokenizer(client_model_args)
        
        return client_model
    
    def _create_client_model_from_global_fedcot(self, client_id: int) -> Any:
        logger.info(f"Initializing client {client_id} from global fedcot model...")
        
        output_dir = Path(self.training_args.output_dir)
        prev_round_dir = output_dir / f"round_{self.current_round - 1}_model"
        prev_global_model = AutoPeftModelForSequenceClassification.from_pretrained(
            str(prev_round_dir),
            num_labels=1,
            ignore_mismatched_sizes=True
        )

        base_model = prev_global_model.merge_and_unload()
        del prev_global_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        client_lora_r = self.client_lora_ranks[client_id]
        client_lora_alpha = client_lora_r

        logger.info(f"Creating client {client_id} with LoRA rank: {client_lora_r}")
        
        client_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=client_lora_r,  
            lora_alpha=client_lora_alpha,  
            lora_dropout=self.model_args.lora_dropout,
            target_modules=["query", "value"],
        )
        
        if hasattr(base_model, "peft_config"):
            delattr(base_model, "peft_config")
        client_model = get_peft_model(base_model, client_config)
        
        return client_model
    
    def load_and_split_data(self, data_args: DataArguments):
        dataset_names = [name.strip() for name in data_args.dataset_names.split(',')]
        dataset_versions = parse_dataset_versions(data_args.dataset_names, data_args.dataset_versions)
        
        if self.federated_args.data_split_method == "iid":
            samples, labels, self.actual_versions = load_dataset_samples(
                data_args.model_output_dir, dataset_names, dataset_versions
            )
            
            self.client_data = DataSplitter.split_data_iid(
                samples, labels, self.federated_args.num_clients
            )
            
        elif self.federated_args.data_split_method == "non_iid_dataset":
            self.client_data, self.actual_versions = DataSplitter.split_data_by_dataset(
                data_args.model_output_dir, dataset_names, dataset_versions, self.federated_args.num_clients
            )
            
        else:
            raise ValueError(f"Unknown data split method: {self.federated_args.data_split_method}")
        
        total_samples = sum(len(client_samples) for client_samples, _ in self.client_data)
        logger.info(f"Distributed {total_samples} training samples across {len(self.client_data)} clients")
    
    def _save_round_model(self, round_num: int) -> str:
        output_dir = Path(self.training_args.output_dir)
        round_model_dir = output_dir / f'round_{round_num}_model'
        round_model_dir.mkdir(parents=True, exist_ok=True)
        
        self.global_model.save_pretrained(round_model_dir)
        logger.info(f"Round {round_num} model saved")
        self.tokenizer.save_pretrained(round_model_dir)
        
        return str(round_model_dir)
    
    def train(self, data_args: DataArguments):
        self.setup_global_model()
        self.load_and_split_data(data_args)
        
        for round_num in range(self.federated_args.num_rounds):
            self.current_round = round_num
        
            logger.info(f"Federated Round {round_num}/{self.federated_args.num_rounds}")
            
            client_models = []
            client_weights = []
            for client_id in range(self.federated_args.num_clients):
                client_samples, client_labels = self.client_data[client_id]
                client_lora_r = self.client_lora_ranks[client_id]
                
                if self.current_round == 0:
                    client_model = self._create_client_model_from_base_fedcot(client_id)
                else:
                    client_model = self._create_client_model_from_global_fedcot(client_id)

                client_training_args = copy.deepcopy(self.training_args)
                client_training_args.num_train_epochs = self.federated_args.client_epochs
                client_training_args.output_dir = f"{self.training_args.output_dir}/temp_client_{client_id}"
                
                client_trainer = RewardModelTrainer(client_model, self.tokenizer, client_training_args)
                
                dataset, _ = prepare_datasets(client_samples, client_labels, self.tokenizer, validation_split=0)
                trainer = client_trainer.create_trainer(dataset)

                trainer.train()

                client_model, num_samples =  client_trainer.model, len(client_samples)
                
                client_models.append(client_model)
                client_weights.append(num_samples)
            
            self.global_model = self.aggregator.aggregate_models(client_models, client_weights)

            # The next round reloads this checkpoint before initializing its clients.
            round_model_path = self._save_round_model(round_num)
            self.saved_models.append((round_num, round_model_path))
            logger.info(f"Round {round_num} model saved to: {round_model_path}")
        
        logger.info("Federated learning training completed!")
    
    def save_global_model(self, output_dir: Path):
        logger.info("Saving Global Model")
        
        if self.federated_args.save_only_final or not self.federated_args.save_all_rounds:
            final_model_dir = output_dir / 'final_model'
            final_model_dir.mkdir(parents=True, exist_ok=True)
            
            self.global_model.save_pretrained(final_model_dir)
            logger.info(f"Final FedCoT model saved to {final_model_dir}")
            
            self.tokenizer.save_pretrained(final_model_dir)
            logger.info(f"Tokenizer saved to {final_model_dir}")
            
            final_round = self.federated_args.num_rounds - 1
            if not any(round_num == final_round for round_num, _ in self.saved_models):
                self.saved_models.append((final_round, str(final_model_dir)))
        
        return self.actual_versions
    
    def get_saved_models_for_evaluation(self) -> List[Tuple[int, str]]:
        if self.federated_args.evaluate_all_rounds:
            return self.saved_models
        else:
            if self.saved_models:
                return [self.saved_models[-1]]
            else:
                return []




