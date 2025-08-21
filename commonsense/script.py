import os

print('enter gpu')
gpu=int(input())

dataset='commonsense_170k'

#model='gpt'
#model='llama'
model='llama3'

lr=5e-4
epoch=3
bs=128
r=16

if model == 'llama2':
    base_model = 'meta-llama/Llama-2-7b-hf'
elif model == 'llama3':
    base_model = 'meta-llama/Meta-Llama-3-8B'
target_modules=["q_proj", "k_proj", "v_proj"]

ratio = 0.03
method = 'oursnewinitfinal'
for seed in [1,2,3]:
    for scale in [8,4]:
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
            f'--lora_alpha {scale * (r**0.5)} '
            f'--prepare_ratio {ratio} '
            f'--target_modules "{",".join(target_modules)}" '
            f'--seed {seed} '
        )
        os.system(cmd)
