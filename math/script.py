import os

print('PID:', os.getpid())

print('enter gpu')
gpu=input()

#model='llama3'
model='gemma'

if model == 'llama3':
    base_model = 'meta-llama/Meta-Llama-3-8B'
    target_modules = 'q_proj k_proj v_proj up_proj down_proj'
elif model == 'gemma':
    base_model = 'google/gemma-7b-it'
    target_modules = 'q_proj k_proj v_proj up_proj down_proj o_proj gate_proj'


bs=64

lr = 2e-4
ratio=0
init='True'
max_steps=-1
epoch=3
for seed in [1,2,3]:
    for r, scale in [(32,4),(64,4)]:
        for method in ['base']:
            alpha = int(scale * (r**0.5))
            os.system(f'CUDA_VISIBLE_DEVICES={gpu} python train_math.py \
                --model_name_or_path {base_model}\
                --data_path ft-training_set/MetaMathQA-395K.json \
                --data_length 10000000 \
                --bf16 True \
                --output_dir ./trained_models/{model}_metamath_epoch{epoch}_r{r}_scale{scale}_lr{lr}_{method}_seed{seed}/\
                --per_device_train_batch_size 4 \
                --per_device_eval_batch_size 4 \
                --gradient_accumulation_steps {bs//4} \
                --evaluation_strategy "no" \
                --save_strategy "no" \
                --learning_rate {lr}\
                --weight_decay 0. \
                --warmup_ratio 0.03 \
                --logging_steps 50 \
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