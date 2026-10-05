import sys
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    DataCollatorWithPadding, TrainingArguments, Trainer,
)
from datasets import Dataset as HFDataset

import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from data_preparation import get_latest_version_dir, parse_dataset_versions

@dataclass
class ModelArguments:
    """Model configuration arguments"""
    model_name: str = field(default="bert-base-uncased")
    max_length: int = field(default=512)
    
    # LoRA configuration
    use_lora: bool = field(default=False, metadata={"help": "Enable LoRA fine-tuning"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    
    random_seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})

@dataclass
class DataArguments:
    """Data configuration arguments"""
    dataset_names: str = field(metadata={"help": "Comma-separated dataset names"})
    model_output_dir: str = field(metadata={"help": "Directory with best-of-n outputs"})
    dataset_versions: Optional[str] = field(default=None, metadata={"help": "Dataset versions"})

@dataclass
class RewardTrainingArguments(TrainingArguments):
    """Training configuration arguments"""
    output_dir: str = field(default="outputs/reward_models")
    per_device_train_batch_size: int = field(default=8)
    learning_rate: float = field(default=2e-5)
    num_train_epochs: float = field(default=3.0)
    weight_decay: float = field(default=0.01)
    save_strategy: str = field(default="epoch")
    validation_split: float = field(default=0.1, metadata={"help": "Validation split ratio"})
    eval_strategy: str = field(default="epoch", metadata={"help": "Evaluation strategy"})
    random_seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})

    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio for learning rate scheduler"})


# ============================================================================
# Data Loading Logic
# ============================================================================

def load_dataset_samples(model_output_dir: str, dataset_names: List[str], 
                        dataset_versions: Dict[str, str]) -> Tuple[List[str], List[int], Dict[str, str]]:
    """Load and process samples from all datasets."""
    all_samples, all_labels = [], []
    actual_versions = {}  
    template = "Q: {question}\nA: {response}"
    
    for dataset_name in dataset_names:
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
            raise FileNotFoundError(f"Data file not found for {dataset_name} version {version}: {data_path}")
        
        logger.info(f"Loading {dataset_name} (requested: {version}, actual: {actual_version}) from {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        for sample in data:
            question = sample['question']
            for candidate in sample['candidates']:
                text = template.format(question=question, response=candidate['response_text'])
                reward = float(candidate.get('reward', 0.0))
                all_samples.append(text)
                all_labels.append(reward)
    
    if not all_samples:
        raise ValueError("No training data found!")
    
    logger.info(f"Loaded {len(all_samples)} samples, {sum(all_labels)} positive ({sum(all_labels)/len(all_samples):.3f})")
    return all_samples, all_labels, actual_versions

# ============================================================================
# Model Setup Functions
# ============================================================================

def setup_model_and_tokenizer(model_args: ModelArguments):
    """Initialize model and tokenizer with optional LoRA"""
    logger.info(f"Loading model: {model_args.model_name}")
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"  
    tokenizer.truncation_side = "left"  
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name, num_labels=1
    )
    
    # Setup LoRA if enabled
    if model_args.use_lora:
        
        logger.info(f"Setting up LoRA: r={model_args.lora_r}, alpha={model_args.lora_r}")
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_r,
            lora_dropout=model_args.lora_dropout,
            target_modules=["query", "value"],

        )
        
        model = get_peft_model(model, lora_config)
    return model, tokenizer

def prepare_datasets(samples: List[str], labels: List[int], tokenizer, max_length: int = 4096, 
                    validation_split: float = 0.1) -> Tuple[HFDataset, HFDataset]:
    """Prepare and tokenize train/validation datasets"""
    full_dataset = HFDataset.from_dict({"text": samples, "labels": labels})
    
    def preprocess_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True,  
            max_length=4096,  
            padding=False  
        )
    
    full_dataset = full_dataset.map(preprocess_function, batched=True, remove_columns=["text"])
    full_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    full_dataset = full_dataset.map(lambda x: {"labels": torch.tensor(x["labels"], dtype=torch.float32)})

    
    if validation_split > 0:
        split_dataset = full_dataset.train_test_split(
            test_size=validation_split, 
            shuffle=True, 
            seed=42
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        train_labels = [sample['labels'] for sample in train_dataset]
        eval_labels = [sample['labels'] for sample in eval_dataset]
        
        train_pos_ratio = sum(train_labels) / len(train_labels)
        eval_pos_ratio = sum(eval_labels) / len(eval_labels)
        
        logger.info(f"Dataset split: {len(train_dataset)} train ({train_pos_ratio:.3f} pos), {len(eval_dataset)} validation ({eval_pos_ratio:.3f} pos)")
        return train_dataset, eval_dataset
    else:
        logger.info(f"No validation split: {len(full_dataset)} train samples")
        return full_dataset, None

# ============================================================================
# Enhanced Trainer with Real-time Loss Logging
# ============================================================================

class SoftRewardTrainer(Trainer):
    """Train a single-logit reward model with labels in the [0, 1] range."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").view(-1, 1).float() 
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

class RewardModelTrainer:
    """Simplified reward model trainer"""
    
    def __init__(self, model, tokenizer, training_args: RewardTrainingArguments):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.trainer = None
    
    
    def create_trainer(self, train_dataset: HFDataset, eval_dataset: HFDataset = None) -> Trainer:
        """Create and configure trainer"""
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        self.trainer = SoftRewardTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  
            data_collator=data_collator,
        )

        return self.trainer
    
def determine_training_method(model_args: ModelArguments) -> str:
    return "lora" if model_args.use_lora else "full"


