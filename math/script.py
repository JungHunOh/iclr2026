import os

print('PID:', os.getpid())

print('enter gpu')
gpu=input()

for model in ['gemma', 'llama3']:

    if model == 'gemma':
        base_model = 'google/gemma-2b'
    elif model == 'llama3':
        base_model = 'meta-llama/Meta-Llama-3-8B'

    bs=128
    mini_bs=8 if model == 'llama3' else 16

    epoch=3
    for seed in [1,2,3]:
        for target_modules in ['q_proj k_proj v_proj down_proj up_proj']:
            target_modules_name = target_modules.replace(' ', '').replace('_proj','')
            for lr in [2e-4]:
                for r, scale in [(32,4)]:
                    #for mode in ['base', 'pissa', 'dora', 'oursinit']:
                    for method in ['base', 'oursinit']:
                        if 'init' in method:
                            max_steps = 50
                        else:
                            max_steps = -1
                        os.system(f'CUDA_VISIBLE_DEVICES={gpu} python train_math.py \
                            --model_name_or_path {base_model}\
                            --data_path ft-training_set/MetaMathQA-40K.json \
                            --data_length 10000000 \
                            --bf16 True \
                            --output_dir ./trained_models/{model}_metamath_epoch{epoch}_bs{bs}_r{r}_scale{scale}_lr{lr}_seed{seed}_{method}_{target_modules_name}/\
                            --per_device_train_batch_size {mini_bs} \
                            --per_device_eval_batch_size 4 \
                            --gradient_accumulation_steps {bs//mini_bs} \
                            --evaluation_strategy "no" \
                            --save_strategy "no" \
                            --learning_rate {lr}\
                            --weight_decay 0. \
                            --warmup_ratio 0 \
                            --logging_steps 20 \
                            --max_steps {max_steps} \
                            --num_train_epochs {epoch} \
                            --lr_scheduler_type "cosine" \
                            --target_modules {target_modules} \
                            --lora_r {r}\
                            --lora_alpha {scale}\
                            --seed {seed}\
                            --lora_dropout 0.05\
                                ')