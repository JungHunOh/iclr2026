import os

gpu = input("Enter GPU ID: ")

#model='llama2'
model = 'llama3'
if model == 'llama3':
    model_name = 'meta-llama/Meta-Llama-3-8B'
elif model == 'llama2':
    model_name = "meta-llama/Llama-2-7b-hf"

dataset = 'alpaca'
#dataset = 'codefeedback'

ratio=0
epoch = 3
lr = 1e-3
bs = 64
r = 32
for scale in [0.5,1,2]:
    method = 'base'
    seed = 1

    output_dir = f"./experiment/{dataset}/{model}_epoch{epoch}_lr{lr}_r{r}_scale{scale}_{method}_seed{seed}"

    # Alpaca finetuning
    os.system(
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"python run_exp.py "
        f"--model_name_or_path {model_name} "
        f"--dataset {dataset} "
        f"--bf16 True "
        f"--output_dir {output_dir} "
        f"--num_train_epochs {epoch} "
        f"--per_device_train_batch_size 4 "
        f"--per_device_eval_batch_size {bs} "
        f"--gradient_accumulation_steps {bs//4} "
        f"--eval_strategy 'no' "
        f"--save_strategy 'no' "
        f"--learning_rate {lr} "
        f"--weight_decay 0. "
        f"--warmup_ratio 0.03 "
        f"--lr_scheduler_type 'cosine' "
        f"--logging_steps 50 "
        f"--tf32 True "
        f"--seed {seed} "
        f"--lora_r {r} "
        f"--lora_alpha {scale * r**0.5} "
        f"--prepare_ratio {ratio} "
    )

