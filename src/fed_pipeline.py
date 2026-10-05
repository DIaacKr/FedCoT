#!/usr/bin/env python3
import os
import json
import sys
import time
import random
import string
import argparse
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Import federated learning components
from federated_reward_model import (
    ModelArguments, DataArguments, RewardTrainingArguments, FederatedArguments,
    FederatedRewardModelTrainer, FedCoTAggregator,
    parse_dataset_versions, determine_training_method,
)

# Import evaluation components
from evaluation import (
    RewardModelSelector,
    run_evaluation, save_results, calculate_summary_statistics
)
from data_preparation import prepare_training_data


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

@dataclass
class FedPipelineConfig:
    """Federated Learning Pipeline Configuration"""
    # Basic configuration
    model_name: str = "Meta-Llama-3-8B-Instruct"
    datasets: str = "bioasq,mmlu,pubmedqa"
    dataset_versions: str = ""
    eval_dataset_versions: str = ""
    reward_model: str = "allenai/longformer-base-4096"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    overwrite_prepared_data: bool = False
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 2e-5
    max_length: int = 512
    
    # LoRA parameters
    lora_r: int = 64
    lora_dropout: float = 0.1
    
    # Federated learning parameters
    num_clients: int = 5
    num_rounds: int = 3
    client_epochs: int = 1
    data_split_method: str = "non_iid_dataset"
    client_lora_r: str = ""
    
    # Evaluation parameters
    eval_batch_size: int = 1024
    eval_max_length: int = 512
    
    # Output control
    output_base_dir: str = "outputs"
    run_evaluation: bool = True
    
    # Random seed
    random_seed: int = 42

    eval_strategy: str = "no"  # Usually no validation during training in federated learning
    validation_split: float = 0.0

    # Round saving and evaluation control
    save_all_rounds: bool = False
    evaluate_all_rounds: bool = False
    save_only_final: bool = True

class FederatedTrainingPipeline:
    """Federated Learning Training and Evaluation Pipeline"""
    
    def __init__(self, config: FedPipelineConfig):
        self.config = config
        self.results = {}

        self._set_all_seeds(config.random_seed)
        
        # Generate unique pipeline ID
        self.pipeline_id = self._generate_pipeline_id()
        
        # Generate output directories
        self.output_dirs = self._generate_output_dirs()
        
        logger.info(f"Pipeline ID: {self.pipeline_id}")

    def _set_all_seeds(self, seed: int):
        """Set all random seeds for reproducibility"""
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        
    
    def _generate_pipeline_id(self) -> str:
        """Generate unique pipeline identifier"""
        pid = os.getpid()
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        microsecond_timestamp = int(time.time() * 1000000) % 1000000
        
        return f"p{pid}_{random_suffix}_{microsecond_timestamp}"
    
    def _generate_output_dirs(self) -> Dict[str, Path]:
        """Generate all output directories (ensure uniqueness)"""
        base_dir = Path(self.config.output_base_dir)
        
        # Special directory structure for federated learning
        training_method = "federated_lora"
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        model_basename = self.config.reward_model.split('/')[-1]
        
        # Federated learning configuration information
        fed_config = f"clients{self.config.num_clients}_rounds{self.config.num_rounds}_{self.config.data_split_method}"
        
        # Format: outputs/reward_models/Meta-Llama-3-8B-Instruct/federated_lora/clients5_rounds3_iid/longformer-base-4096-2506081234_p1234_abc5_567890
        model_output_dir = (base_dir / "reward_models" / self.config.model_name / 
                           training_method / fed_config / f"{model_basename}-{timestamp}_{self.pipeline_id}")
        
        # Evaluation directory
        eval_base_dir = base_dir / "evaluations" / self.config.model_name / "federated_reward_model"
        eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_output_dir = eval_base_dir / f"{eval_timestamp}_{self.pipeline_id}"
        
        return {
            'model': model_output_dir,
            'evaluation': eval_output_dir,
            'data': base_dir / "best_of_n" / self.config.model_name
        }
    
    def _convert_to_training_args(self) -> Tuple[ModelArguments, DataArguments, RewardTrainingArguments, FederatedArguments]:
        """Convert configuration to training arguments"""
        model_args = ModelArguments(
            model_name=self.config.reward_model,
            max_length=self.config.max_length,
            use_lora=True,
            lora_r=self.config.lora_r,
            lora_dropout=self.config.lora_dropout,
            random_seed=self.config.random_seed
        )
        
        data_args = DataArguments(
            dataset_names=self.config.datasets,
            model_output_dir=str(self.output_dirs['data']),
            dataset_versions=self.config.dataset_versions
        )
        
        training_args = RewardTrainingArguments(
            output_dir=str(self.output_dirs['model']),
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.client_epochs,  # Client training rounds
            save_strategy="no",
            logging_steps=0.1,
            bf16=True,
            report_to=[],
            eval_strategy=self.config.eval_strategy,
            validation_split=self.config.validation_split,
        )
        
        federated_args = FederatedArguments(
            num_clients=self.config.num_clients,
            num_rounds=self.config.num_rounds,
            client_epochs=self.config.client_epochs,
            data_split_method=self.config.data_split_method,
            client_lora_r=self.config.client_lora_r,
            save_all_rounds=self.config.save_all_rounds,
            evaluate_all_rounds=self.config.evaluate_all_rounds,
            save_only_final=self.config.save_only_final
        )
        
        return model_args, data_args, training_args, federated_args
    
    def run_training(self) -> Dict[str, Any]:
        logger.info(f"Training FedCoT reward model: {self.output_dirs['model']}")
        
        # Convert configuration
        model_args, data_args, training_args, federated_args = self._convert_to_training_args()
        
        # Create output directory
        self.output_dirs['model'].mkdir(parents=True, exist_ok=True)
        
        # Create aggregator
        if federated_args.client_lora_r:
            client_ranks = [int(rank.strip()) for rank in federated_args.client_lora_r.split(',')]
        else:
            client_ranks = [model_args.lora_r] * federated_args.num_clients
        aggregator = FedCoTAggregator(client_lora_ranks=client_ranks)
        
        # Create federated learning trainer
        fed_trainer = FederatedRewardModelTrainer(
            model_args, training_args, federated_args, aggregator
        )
        
        # Execute federated learning training
        fed_trainer.train(data_args)
        
        # Save global model and metadata
        actual_versions = fed_trainer.save_global_model(self.output_dirs['model'])
        
        # Get list of models to evaluate
        saved_models = fed_trainer.get_saved_models_for_evaluation()
        
        training_results = {
            'model_path': str(self.output_dirs['model']),  # Base directory
            'model_name': f"{self.output_dirs['model'].name}",
            'training_method': f"federated_{determine_training_method(model_args)}",
            'datasets': [name.strip() for name in data_args.dataset_names.split(',')],
            'num_clients': federated_args.num_clients,
            'num_rounds': federated_args.num_rounds,
            'data_split_method': federated_args.data_split_method,
            'aggregation_method': 'fedcot',
            'dataset_versions': actual_versions,
            'saved_models': saved_models,  # [(round_num, model_path), ...]
            'evaluate_all_rounds': federated_args.evaluate_all_rounds,
            'final_model': saved_models[-1][1] if saved_models else None
        }
        
        self.results['training'] = training_results
        return training_results
    
    def run_evaluation(self) -> Dict[str, Any]:
        training = self.results['training']
        saved_models = training['saved_models']
        evaluate_all_rounds = training['evaluate_all_rounds']
        self.output_dirs['evaluation'].mkdir(parents=True, exist_ok=True)

        if evaluate_all_rounds:
            models = [
                {
                    'path': model_path,
                    'name': f"round_{round_num + 1}_model",
                    'round': round_num + 1
                }
                for round_num, model_path in saved_models
            ]
        else:
            final_round, final_model_path = saved_models[-1]
            models = [{
                'path': final_model_path,
                'name': "final_model",
                'round': final_round + 1,
            }]

        dataset_names = [name.strip() for name in self.config.datasets.split(',')]
        dataset_versions = parse_dataset_versions(self.config.datasets, self.config.eval_dataset_versions)
        evaluation_results = {}
        for model_info in models:
            selector = RewardModelSelector(
                model_path=model_info['path'],
                batch_size=self.config.eval_batch_size,
                max_length=self.config.eval_max_length,
                random_seed=self.config.random_seed,
            )
            results, actual_versions = run_evaluation(
                str(self.output_dirs['data']), selector, dataset_names, dataset_versions
            )
            results = calculate_summary_statistics(results)
            model_eval_dir = self.output_dirs['evaluation'] / model_info['name']
            model_eval_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = model_eval_dir / f"evaluation_{timestamp}.json"
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'model_path': model_info['path'],
                'model_name': model_info['name'],
                'round': model_info['round'],
                'dataset_names': dataset_names,
                'dataset_versions_requested': dataset_versions,
                'dataset_versions_used': actual_versions,
                'selector_info': selector.get_method_info(),
            }
            save_results(results, output_path, metadata)
            evaluation_results[model_info['name']] = {
                'results': results,
                'metadata': metadata,
                'output_path': str(output_path),
            }

        self.results['evaluation'] = evaluation_results
        return evaluation_results
    
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        dataset_names = [name.strip() for name in self.config.datasets.split(",") if name.strip()]
        dataset_versions = parse_dataset_versions(self.config.datasets, self.config.dataset_versions)
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        # Preparation used to run in a separate process; keep training RNGs unchanged.
        with torch.random.fork_rng():
            try:
                prepare_training_data(
                    self.output_dirs['data'],
                    dataset_names,
                    dataset_versions,
                    self.config.nli_model,
                    self.config.random_seed,
                    self.config.overwrite_prepared_data,
                )
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        self.run_training()
        if self.config.run_evaluation:
            self.run_evaluation()
        return self.results
    
def main():
    """Command line interface"""
    
    
    parser = argparse.ArgumentParser(description='Federated Learning Training and Evaluation Pipeline')
    
    # Basic configuration
    parser.add_argument('--model_name', default="Meta-Llama-3-8B-Instruct")
    parser.add_argument('--datasets', default="bioasq,mmlu,pubmedqa")
    parser.add_argument('--dataset_versions', default="")
    parser.add_argument('--eval_dataset_versions', default="")
    parser.add_argument('--reward_model', default="allenai/longformer-base-4096")
    parser.add_argument('--nli_model', default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument('--overwrite_prepared_data', action='store_true')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=512)
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    
    # Federated learning parameters
    parser.add_argument('--num_clients', type=int, default=5, help="Number of clients")
    parser.add_argument('--num_rounds', type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument('--client_epochs', type=int, default=1, help="Number of client local training rounds")
    parser.add_argument('--data_split_method', default="iid", choices=["iid", "non_iid_dataset"],
                       help="Data splitting method")
    parser.add_argument('--client_lora_r', default="", help="Client LoRA rank list (comma-separated, used by FedCoT)")
    
    # Evaluation parameters
    parser.add_argument('--eval_batch_size', type=int, default=1024)
    parser.add_argument('--eval_max_length', type=int, default=512)
    
    # Control parameters
    parser.add_argument('--skip_evaluation', action='store_true', help="Skip evaluation step")
    
    # Round saving and evaluation control
    parser.add_argument('--save_all_rounds', action='store_true', help="Save global models from all rounds")
    parser.add_argument('--evaluate_all_rounds', action='store_true', help="Evaluate global models from all rounds")
    parser.add_argument('--save_only_final', dest='save_only_final', action='store_true', help="Only save global model from final round")
    parser.add_argument('--no_save_only_final', dest='save_only_final', action='store_false', help="Save global models from all rounds")
    parser.set_defaults(save_only_final=True)
    
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    
    logger.info(f"FedCoT training with {args.num_clients} clients and {args.num_rounds} rounds")
    
    # Create configuration
    config = FedPipelineConfig(
        model_name=args.model_name,
        datasets=args.datasets,
        dataset_versions=args.dataset_versions,
        eval_dataset_versions=args.eval_dataset_versions,
        reward_model=args.reward_model,
        nli_model=args.nli_model,
        overwrite_prepared_data=args.overwrite_prepared_data,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        client_epochs=args.client_epochs,
        data_split_method=args.data_split_method,
        client_lora_r=args.client_lora_r,
        eval_batch_size=args.eval_batch_size,
        eval_max_length=args.eval_max_length,
        run_evaluation=not args.skip_evaluation,
        save_all_rounds=args.save_all_rounds,
        evaluate_all_rounds=args.evaluate_all_rounds,
        save_only_final=args.save_only_final,
        random_seed=args.random_seed
    )
    
    # Create and run federated learning pipeline
    pipeline = FederatedTrainingPipeline(config)
    pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()

