import os

print('PID:', os.getpid())

print('enter gpu')
gpu=input()

#model='gpt'
#model='llama'
#model='llama2'
for model in ['llama3', 'llama2']:

    if model == 'gpt':
        base_model = 'EleutherAI/gpt-j-6b'
    elif model == 'llama':
        base_model = 'yahma/llama-7b-hf'
    elif model == 'llama2':
        base_model = "meta-llama/Llama-2-7b-hf"
    elif model == 'llama3':
        base_model = 'meta-llama/Meta-Llama-3-8B'

    bs=128

    ratio=0.03
    init='True'
    max_steps=-1
    epoch=3
    for seed in [1]:
        for target_modules in ['q_proj k_proj v_proj']:
            target_modules_name = target_modules.replace(' ', '').replace('_proj','')
            for lr in [5e-4]:
                for r, scale in [(32,8),(32,4)]:
                    for method in ['oursnewinitfinal']:
                        alpha = int(scale * (r**0.5))
                        os.system(f'CUDA_VISIBLE_DEVICES={gpu} python train_math.py \
                            --model_name_or_path {base_model}\
                            --data_path ft-training_set/MetaMathQA-40K.json \
                            --data_length 10000000 \
                            --bf16 True \
                            --output_dir ./trained_models/{model}_metamath_epoch{epoch}_bs{bs}_r{r}_scale{scale}_lr{lr}_seed{seed}_{method}_{target_modules_name}/\
                            --per_device_train_batch_size 8 \
                            --per_device_eval_batch_size 4 \
                            --gradient_accumulation_steps {bs//8} \
                            --evaluation_strategy "no" \
                            --save_strategy "no" \
                            --learning_rate {lr}\
                            --weight_decay 0. \
                            --warmup_ratio 0 \
                            --logging_steps 10 \
                            --max_steps {max_steps} \
                            --num_train_epochs {epoch} \
                            --lr_scheduler_type "cosine" \
                            --target_modules {target_modules} \
                            --lora_r {r}\
                            --lora_alpha {alpha}\
                            --seed {seed}\
                            --lora_dropout 0.05\
                            --lora_init {init}\
                            --prepare_ratio {ratio}\
                                ')