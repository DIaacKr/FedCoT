import json
import os
from pathlib import Path
from datetime import datetime
import click
import numpy as np
import openai
import tqdm
from omegaconf import OmegaConf
from sglang.utils import launch_server_cmd, terminate_process, wait_for_server
from transformers import AutoTokenizer
from jinja2 import Template
import re

from dataset_loader import get_datasets

from prompt_loader import load_prompts
from answer_extractor import extract_answer, match_choice_answer, compute_reward

@click.command()
@click.option("--config_path", "-c", type=str, default="configs/best_of_n.yaml")
@click.option("--update_config_by_dotlist", "-u", type=str, default=None)
@click.option("--dataset_names", type=str, required=True, help="Comma-separated dataset names (e.g., mmlu,medqa,bioasq)")
@click.option("--num_candidates", type=int, default=8, help="Number of candidates to generate")
@click.option("--modes", type=str, default="train,test", help="Comma-separated modes to process (train,test)")
@click.option("--overwrite", is_flag=True, help="Overwrite existing results")
@click.option("--enable_cot", is_flag=True, help="Enable Chain-of-Thought prompting")
def main(
    config_path,
    update_config_by_dotlist,
    dataset_names,
    num_candidates,
    modes,
    overwrite,
    enable_cot
):
    config = OmegaConf.load(config_path)
    if update_config_by_dotlist is not None:
        update_config_by_dotlist = update_config_by_dotlist.split(",")
        update_config_by_dotlist = OmegaConf.from_dotlist(update_config_by_dotlist)
        config = OmegaConf.merge(config, update_config_by_dotlist)
    
    # Override config with CLI args
    config.dataset_names = dataset_names
    config.num_candidates = num_candidates
    config.modes = modes
    config.overwrite = overwrite
    if enable_cot:
        config.enable_cot = enable_cot
    
    # Extract model name from path for output directory
    model_name = Path(config.model_path).name
    
    # Setup base output directory with model name
    base_output_dir = Path(config.output_dir) / model_name
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for this execution
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add execution metadata to config for archival
    config_with_metadata = OmegaConf.create(dict(config))
    config_with_metadata.execution_metadata = {
        'timestamp': timestamp,
        'original_config_path': config_path,
        'cli_args': {
            'dataset_names': dataset_names,
            'num_candidates': num_candidates,
            'modes': modes,
            'overwrite': overwrite,
            'enable_cot': enable_cot
        },
        'update_config_by_dotlist': update_config_by_dotlist
    }
    
    # Save current config to main directory (overwrite latest)
    current_config_file = base_output_dir / "config.yaml"
    with open(current_config_file, "w") as f:
        OmegaConf.save(config_with_metadata, f)
    
    # Start SGLang and connect the OpenAI-compatible client.
    sglang_server = SGLangServer(config)
    sglang_server.start()

    try:
        client = openai.Client(
            base_url=f"http://127.0.0.1:{config.port}/v1",
            api_key="EMPTY",
        )
        tokenizer_path = config.tokenizer_path or config.model_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, padding_side="left"
        )
        if not getattr(tokenizer, "chat_template", None):
            model_path_lower = config.model_path.lower()
            if "llama" in model_path_lower:
                tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
            elif "qwen" in model_path_lower:
                tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|im_start|>' + message['role'] + '\n'+ message['content'] | trim + '<|im_end|>\n' %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
            else:
                tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"

        processor = BestOfNProcessor(
            config, client, tokenizer, Template(tokenizer.chat_template), base_output_dir
        )
        dataset_list = [name.strip() for name in dataset_names.split(',')]
        mode_list = [mode.strip() for mode in modes.split(',')]
        for dataset_name in dataset_list:
            for mode in mode_list:
                processor.process_single_dataset_mode(dataset_name, mode)
    finally:
        sglang_server.terminate()


def prepare_version_dir(base_dir, mkdir=False):
    """Create a versioned directory to avoid conflicts"""
    base_dir = Path(base_dir)
    
    existing_versions = []
    if base_dir.exists():
        for d in base_dir.iterdir():
            if d.is_dir() and d.name.startswith("version_"):
                try:
                    version_num = int(d.name.split("_")[1])
                    existing_versions.append(version_num)
                except (IndexError, ValueError):
                    continue
    
    if len(existing_versions) == 0:
        next_version = 1
    else:
        next_version = max(existing_versions) + 1
    
    version_dir = base_dir / f"version_{next_version}"
    
    if mkdir:
        version_dir.mkdir(parents=True, exist_ok=True)
    
    return version_dir, next_version

class BestOfNProcessor:
    def __init__(self, config, client, tokenizer, template, base_output_dir):
        self.config = config
        self.client = client
        self.tokenizer = tokenizer
        self.template = template
        self.base_output_dir = base_output_dir

        self.extract_answer = extract_answer
        self.match_choice_answer = match_choice_answer
        self.compute_reward_func = compute_reward

        self.load_prompts = load_prompts
        
    def process_single_dataset_mode(self, dataset_name, mode):
        """Process a single dataset-mode combination"""
        train_dataset, val_dataset, test_dataset = get_datasets(self.config.seed, dataset_name)
        
        if mode == 'train':
            dataset = train_dataset
        else:
            dataset = test_dataset
        
        if self.config.limit > 0:
            dataset = dataset[:self.config.limit]
        
        # Setup output directory for this dataset-mode
        base_dataset_dir = self.base_output_dir / f"{dataset_name}_best_of_n" / mode
        output_file = base_dataset_dir / f"{mode}_best_of_n.json"
        
        if output_file.exists():
            if getattr(self.config, 'overwrite', False):
                output_dir = base_dataset_dir
            else:
                version_dir, version_num = prepare_version_dir(base_dataset_dir, mkdir=True)
                output_dir = version_dir
                output_file = output_dir / f"{mode}_best_of_n.json"
        else:
            output_dir = base_dataset_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_data = []
        for sample_idx, item in enumerate(dataset):
            for candidate_idx in range(self.config.num_candidates):
                prompt_data = {
                    'sample_idx': sample_idx,
                    'candidate_idx': candidate_idx,
                    'question': item['question'],
                    'answer_string': item['answer_string'],
                    'answer_letter': item['answer_letter'],
                    'answer_idx': item['answer_idx'],
                    'options': item['options'],
                    'input_str': self.prepare_prompt(item, dataset_name)
                }
                all_data.append(prompt_data)
        
        print(f"Total prompts to process: {len(all_data)}")
        
        result_dict_list = []
        for start_idx in tqdm.tqdm(
            range(0, len(all_data), self.config.batch_size), 
            desc=f"Processing {dataset_name}-{mode} batches"
        ):
            batch_data = all_data[start_idx:start_idx + self.config.batch_size]
            prompts = [item["input_str"] for item in batch_data]
            
            responses = self.call_model_batch(prompts)
            
            for item, response in zip(batch_data, responses):
                response_text = response["response_text"]
                finish_reason = response["finish_reason"]
                num_gen_tokens = response["num_gen_tokens"]
                
                extracted_answer = self.extract_and_match_answer(response_text, item['options'])
                
                reward = self.compute_reward(extracted_answer, item)
                is_correct = reward == 1.0
                
                result_item = {
                    'sample_idx': item['sample_idx'],
                    'candidate_idx': item['candidate_idx'],
                    'question': item['question'],
                    'answer_string': item['answer_string'],
                    'answer_letter': item['answer_letter'],
                    'answer_idx': item['answer_idx'],
                    'options': item['options'],
                    'response_text': response_text,
                    'finish_reason': finish_reason,
                    'num_gen_tokens': num_gen_tokens,
                    'extracted_answer': extracted_answer,
                    'reward': reward,
                    'is_correct': is_correct,
                    'dataset_name': dataset_name,
                    'mode': mode
                }
                
                result_dict_list.append(result_item)
        
        processed_data = self.reorganize_results(result_dict_list, len(dataset), mode)
    
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        print(f"Saved {len(processed_data)} processed samples to {output_file}")
        
        self.save_statistics(processed_data, output_dir, dataset_name, mode)
    
    def reorganize_results(self, result_dict_list, num_samples, mode):
        processed_data = []
        
        for sample_idx in range(num_samples):
            sample_results = [r for r in result_dict_list if r['sample_idx'] == sample_idx]
            sample_results.sort(key=lambda x: x['candidate_idx'])
            
            if not sample_results:
                continue
            
            candidates = []
            for result in sample_results:
                candidate = {
                    'candidate_id': result['candidate_idx'],
                    'response_text': result['response_text'],
                    'finish_reason': result['finish_reason'],
                    'num_gen_tokens': result['num_gen_tokens'],
                    'extracted_answer': result['extracted_answer'],
                    'reward': result['reward'],
                    'is_correct': result['is_correct']
                }
                
                candidates.append(candidate)
            
            sample_result = {
                'question': sample_results[0]['question'],
                'answer_string': sample_results[0]['answer_string'],
                'answer_letter': sample_results[0]['answer_letter'],
                'answer_idx': sample_results[0]['answer_idx'],
                'options': sample_results[0]['options'],
                'candidates': candidates,
                'num_candidates': len(candidates),
                'dataset_name': sample_results[0]['dataset_name'],
                'mode': sample_results[0]['mode']
            }
            
            num_correct = sum(1 for c in candidates if c.get('is_correct', False))
            sample_result.update({
                'num_correct': num_correct,
                'accuracy': num_correct / len(candidates) if len(candidates) > 0 else 0,
                'best_candidate_id': max(range(len(candidates)), 
                                       key=lambda i: candidates[i]['reward']) if candidates else 0,
                'worst_candidate_id': min(range(len(candidates)), 
                                        key=lambda i: candidates[i]['reward']) if candidates else 0
            })
            
            processed_data.append(sample_result)
        
        return processed_data
    
    def prepare_prompt(self, item, dataset_name):
        """Prepare prompt for generation with optional CoT"""
        enable_cot = getattr(self.config, 'enable_cot', False)
        
        if enable_cot:
            cot_prompt = self.load_prompts(dataset_name)

            query_prompt = cot_prompt + "\n\nQ: " + item['question']
        else:
            query_prompt = item['question']
        
        if getattr(self.config, 'prefix_prompt', ''):
            query_prompt = self.config.prefix_prompt + "\n" + query_prompt
        if getattr(self.config, 'suffix_prompt', ''):
            query_prompt = query_prompt + "\n" + self.config.suffix_prompt
        
        return query_prompt
    
    def call_model_batch(self, prompts):
        formatted_prompts = [
            self.template.render(
                messages=[{"role": "user", "content": prompt}],
                bos_token=getattr(self.tokenizer, "bos_token", ""),
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        
        if getattr(self.config, 'enable_diversity', True):
            temp_range = getattr(self.config, 'temp_range')
            top_p_range = getattr(self.config, 'top_p_range')
            
            temperature = np.random.uniform(temp_range[0], temp_range[1])
            top_p = np.random.uniform(top_p_range[0], top_p_range[1])
        else:
            temperature = self.config.temperature
            top_p = getattr(self.config, 'top_p', 0.9)
        
        # Call API
        response = self.client.completions.create(
            model="default",
            prompt=formatted_prompts,
            temperature=temperature,
            top_p=top_p,
            max_tokens=self.config.max_new_tokens,
            frequency_penalty=self.config.frequency_penalty,
            timeout=self.config.timeout,
        )
        
        # Process responses
        results = []
        for _response in response.choices:
            results.append({
                "response_text": _response.text,
                "finish_reason": _response.finish_reason,
                "num_gen_tokens": len(
                    self.tokenizer.encode(_response.text, add_special_tokens=False)
                ),
            })
        
        return results
    
    def extract_and_match_answer(self, response_text, options_dict):
        """Extract answer and match with options"""
        # Step 1: Extract raw answer
        raw_extracted = self.extract_answer(response_text)
        
        # Step 2: Match with choice options
        matched_answer = self.match_choice_answer(raw_extracted, options_dict)
        
        return matched_answer
    
    def compute_reward(self, extracted_answer, ground_truth_item):
        """Compute binary reward for training"""
        return self.compute_reward_func(extracted_answer, ground_truth_item)
    
    def save_statistics(self, processed_data, output_dir, dataset_name, mode):
        """Save processing statistics"""
        stats = {
            'total_samples': len(processed_data),
            'num_candidates_per_sample': self.config.num_candidates,
            'mode': mode,
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat()
        }
        
        if processed_data:
            pass_at_1_count = 0
            pass_at_k_count = 0
            
            for item in processed_data:
                candidates = item['candidates']
                if not candidates:
                    continue
                
                first_correct = candidates[0].get('is_correct', False)
                any_correct = any(c.get('is_correct', False) for c in candidates)
                
                if first_correct:
                    pass_at_1_count += 1
                if any_correct:
                    pass_at_k_count += 1
            
            total_samples = len(processed_data)
            pass_at_1 = pass_at_1_count / total_samples if total_samples > 0 else 0
            pass_at_k = pass_at_k_count / total_samples if total_samples > 0 else 0
            
            stats.update({
                'pass_at_1': pass_at_1,
                f'pass_at_{self.config.num_candidates}': pass_at_k,
                'pass_at_1_count': pass_at_1_count,
                f'pass_at_{self.config.num_candidates}_count': pass_at_k_count,
                'improvement_potential': pass_at_k - pass_at_1,  
            })
        
        total_candidates = sum(len(item['candidates']) for item in processed_data)
        total_correct = sum(item['num_correct'] for item in processed_data)
        
        finish_reason_counts = {}
        length_limit_count = 0
        total_gen_tokens = 0
        max_gen_tokens = 0
        min_gen_tokens = float('inf')
        
        for item in processed_data:
            for candidate in item['candidates']:
                finish_reason = candidate.get('finish_reason', 'unknown')
                finish_reason_counts[finish_reason] = finish_reason_counts.get(finish_reason, 0) + 1
                
                if finish_reason == 'length':
                    length_limit_count += 1
                
                num_tokens = candidate.get('num_gen_tokens', 0)
                total_gen_tokens += num_tokens
                max_gen_tokens = max(max_gen_tokens, num_tokens)
                if num_tokens > 0:  
                    min_gen_tokens = min(min_gen_tokens, num_tokens)
        
        if min_gen_tokens == float('inf'):
            min_gen_tokens = 0
        
        avg_gen_tokens = total_gen_tokens / total_candidates if total_candidates > 0 else 0
        length_limit_ratio = length_limit_count / total_candidates if total_candidates > 0 else 0
        
        stats.update({
            'total_candidates': total_candidates,
            'total_correct': total_correct,
            'overall_accuracy': total_correct / total_candidates if total_candidates > 0 else 0,
            'avg_accuracy_per_sample': np.mean([item['accuracy'] for item in processed_data]),
            'samples_with_correct_answers': sum(1 for item in processed_data if item['num_correct'] > 0),
            'samples_with_all_correct': sum(1 for item in processed_data if item['num_correct'] == self.config.num_candidates),
            
            'generation_stats': {
                'finish_reason_distribution': finish_reason_counts,
                'length_limit_count': length_limit_count,
                'length_limit_ratio': length_limit_ratio,
                'avg_gen_tokens': avg_gen_tokens,
                'max_gen_tokens': max_gen_tokens,
                'min_gen_tokens': min_gen_tokens,
                'configured_max_tokens': self.config.max_new_tokens
            }
        })
        
        stats_file = output_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats

    def _check_answer_correct(self, extracted_answer, sample_item):
        ground_truth = {
            'answer_letter': sample_item['answer_letter'],
            'answer_string': sample_item['answer_string']
        }
        reward = self.compute_reward(extracted_answer, ground_truth)
        return reward == 1.0


class SGLangServer:
    def __init__(self, config):
        self.config = config
        self.model_path = config.model_path
        self.port = config.port
        self.dp = config.dp
        self.tp = config.tp
        self.mem_fraction_static = config.mem_fraction_static
        self.seed = config.seed
        self.log_level = config.log_level
        self.server_process = None

    def start(self):
        server_process, port = launch_server_cmd(
            f"""
        python -m sglang.launch_server \
        --model-path {self.model_path} \
        --mem-fraction-static {self.mem_fraction_static} \
        --dp {self.dp} \
        --tp {self.tp} \
        --random-seed {self.seed} \
        --log-level {self.log_level}
        """,
            port=self.port,
        )
        
        if port != self.port:
            self.terminate()
            raise ValueError(f"Port mismatch: return {port} != set {self.port}")

        wait_for_server(f"http://localhost:{port}")
        print(f"SGLang server started at port {port}")
        self.server_process = server_process

    def terminate(self):
        if self.server_process is not None:
            terminate_process(self.server_process)
            print("SGLang server terminated")


if __name__ == "__main__":
    main()

