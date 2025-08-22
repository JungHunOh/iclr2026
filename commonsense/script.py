import os

print('enter gpu')
gpu=int(input())

dataset='commonsense_170k'


for model in ['llama3', 'gemma']:

    if model == 'gemma':
        base_model = 'google/gemma-2b'
    elif model == 'llama3':
        base_model = 'meta-llama/Meta-Llama-3-8B'
    
    lr=5e-4
    epoch=3
    bs=128
    scale=4

    target_modules=["q_proj", "k_proj", "v_proj"]

    #for mode in ['base', 'pissa', 'dora', 'oursinit']:
    for method in ['base']:
        for seed in [1,2,3]:
            for r in [8,16,32]:
                if 'init' in method:
                    max_steps = 50
                else:
                    max_steps = -1
                cmd = (
                    f'CUDA_VISIBLE_DEVICES={gpu} python finetune.py '
                    f'--base_model {base_model} '
                    f'--data_path ./ft-training_set/{dataset}.json '
                    f'--output_dir ./trained_models/{model}_{dataset}_lr{lr}_epoch{epoch}_bs{bs}_r{r}_scale{scale}_{method}_seed{seed}/ '
                    f'--batch_size {bs} '
                    f'--micro_batch_size 16 '
                    f'--num_epochs {epoch} '
                    f'--learning_rate {lr} '
                    f'--cutoff_len 256 '
                    f'--val_set_size 0 '
                    f'--adapter_name lora '
                    f'--lora_r {r} '
                    f'--lora_alpha {scale} '
                    f'--target_modules "{",".join(target_modules)}" '
                    f'--seed {seed} '
                    f'--max_steps {max_steps} '
                )
                os.system(cmd)
