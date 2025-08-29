import os

gpu = input("Enter GPU ID: ")

for model in ['gemma','llama3']:

    if model == 'gemma':
        base_model = 'google/gemma-2b'
    elif model == 'llama3':
        base_model = 'meta-llama/Meta-Llama-3-8B'

    #dataset = 'alpaca'
    dataset = 'codefeedback'

    epoch = 3
    seed = 1
    bs = 128
    mini_bs=4 if model == 'llama3' else 8
    scale = 4
    for lr in [1e-4,2e-4]:
        for r in [32]:
            for target_modules in ['q_proj k_proj v_proj down_proj up_proj']:
                target_modules_name = target_modules.replace(' ', '').replace('_proj','')
                
                for method in ['oursinitone', 'base']:
                #for method in ['base']:
                    if 'init' in method:
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
                        f"--logging_steps 50 "
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
