import json
import torch
import argparse
import math
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Constants ---
DATASET_NAME = "tatsu-lab/alpaca_eval"
DATASET_CONFIG = "alpaca_eval"

def apply_chat_template(tokenizer, instruction):
    """
    Applies the correct chat template based on the tokenizer.
    Supports Llama 3 and Gemma chat templates.
    """
    # Llama 3 uses a specific chat template format.
    if "Llama-3" in tokenizer.name_or_path or "llama3" in tokenizer.name_or_path.lower():
        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Gemma uses a different turn-based format.
    elif "gemma" in tokenizer.name_or_path.lower():
        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback for other models, may need adjustment.
        print("Warning: Model type not explicitly recognized. Using a generic user/assistant template.")
        prompt = f"User: {instruction}\nAssistant:"

    return prompt


def eval(model, tokenizer, dir):
    save_dir = dir+'/alpaca_eval_outputs.json'
    model.eval()

    # --- 2. Load Dataset ---
    print("Loading AlpacaEval dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)["eval"]

    # --- 3. Generate Outputs ---
    model_outputs = []
    print(f"\nStarting generation on {len(dataset)} examples...")

    if 'gemma' in dir:
        batch_size = 300
    else:
        batch_size = 50
    num_batches = math.ceil(len(dataset) / batch_size)
    print(f"\nStarting generation on {len(dataset)} examples in {num_batches} batches...")

    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Batches"):
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        batch_examples = [dataset[j] for j in batch_indices]
        instruction_batch = [ex['instruction'] for ex in batch_examples]

        inputs = tokenizer(instruction_batch, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,  # Use greedy decoding for reproducibility
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the batch of generated texts
        # We need to slice the prompt out of the generated tokens
        prompt_lengths = inputs['input_ids'].shape[1]
        decoded_texts = tokenizer.batch_decode(outputs[:, prompt_lengths:], skip_special_tokens=True)

        # Append results for the current batch
        for instruction, response_text in zip(instruction_batch, decoded_texts):
            model_outputs.append({
                "instruction": instruction,
                "output": response_text.strip(),
                "generator": 'my_model'
            })

    # --- 4. Save Outputs ---
    print(f"\nSaving outputs to {save_dir}...")
    with open(save_dir, "w") as f:
        json.dump(model_outputs, f, indent=4)

    print("--- Generation Complete! ---")
    print(f"Your outputs are ready at: ./{save_dir}")
    print("\nNext step: Run the alpaca_eval command on this file.")

    import os
    os.system(f"export OPENAI_API_KEY=tgp_v1_cPnYwhg9IUNxfvC-YhGvs8wzySBYED6B9SeH9xVgZIc && alpaca_eval --model_outputs {save_dir} --annotators_config alpaca_eval_llama3_70b_fn --output_path {dir}")
