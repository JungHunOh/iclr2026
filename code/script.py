import os

gpu = input("Enter GPU ID: ")

for model in ['gemma','llama3','llama2']:

    if model == 'gemma':
        base_model = 'google/gemma-2b'
    elif model == 'llama3':
        base_model = 'meta-llama/Meta-Llama-3-8B'
    elif model == 'llama2':
        base_model = 'meta-llama/Llama-2-7b-hf'

    #dataset = 'alpaca'
    dataset = 'codefeedback'

    epoch = 1
    bs = 32
    if model == 'llama3':
        mini_bs = 4
        lr = 1e-4
    elif model == 'gemma':
        mini_bs = 16
        lr = 2e-4
    elif model == 'llama2':
        mini_bs = 8
        lr = 2e-4

    scale = 4
    for seed in [1,2,3]:
        for r in [8,32]:
            for target_modules in ['q_proj k_proj v_proj down_proj up_proj gate_proj o_proj']:
                target_modules_name = target_modules.replace(' ', '').replace('_proj','')
                
                #['base', 'pissa', 'dora', 'odlora', 'norslora', 'lora+']
                for method in ['base', 'pissa', 'dora', 'odlora', 'norslora', 'lora+']:
                    if 'odlora' in method:
                        max_steps = 50
                    else:
                        max_steps = -1

                    output_dir = f"./experiment/{dataset}/{model}_epoch{epoch}_bs{bs}_lr{lr}_r{r}_scale{scale}_seed{seed}_{method}_{target_modules_name}"
                    os.makedirs(output_dir, exist_ok=True)
                    # Alpaca finetuning
                    os.system(
                        f"CUDA_VISIBLE_DEVICES={gpu} "
                        f"python run_exp.py "
                        f"--model_name_or_path {base_model} "
                        f"--dataset {dataset} "
                        f"--bf16 True "
                        f"--output_dir {output_dir} "
                        f"--num_train_epochs {epoch} "
                        f"--per_device_train_batch_size {mini_bs} "
                        f"--per_device_eval_batch_size {bs} "
                        f"--gradient_accumulation_steps {bs//mini_bs} "
                        f"--eval_strategy 'no' "
                        f"--save_strategy 'no' "
                        f"--learning_rate {lr} "
                        f"--weight_decay 0. "
                        f"--warmup_ratio 0 "
                        f"--lr_scheduler_type 'cosine' "
                        f"--logging_steps 20 "
                        f"--tf32 True "
                        f"--seed {seed} "
                        f"--lora_r {r} "
                        f"--lora_alpha {scale} "
                        f"--target_modules {target_modules} "
                        f"--max_steps {max_steps} "
                        f"| tee {output_dir}/log.txt"
                    )
                    print(output_dir)
input()
