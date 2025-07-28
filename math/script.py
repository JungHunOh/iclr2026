import os

print('PID:', os.getpid())

print('enter gpu')
gpu=input()

#model='gpt'
#model='llama'
model='llama3'

if model == 'gpt':
    base_model = 'EleutherAI/gpt-j-6b'
elif model == 'llama':
    base_model = 'yahma/llama-7b-hf'
elif model == 'llama3':
    base_model = 'meta-llama/Meta-Llama-3-8B'

bs=32

lr = 1e-4
init='True'
for seed in [1]:
    for r, ratio, scale, epoch in [(4,0,4,4),(16,0,4,4),(4,0,8,4),(16,0,8,4)]:
        alpha = int(scale * (r**0.5))
        os.system(f'CUDA_VISIBLE_DEVICES={gpu} python finetune.py \
            --model_name_or_path {base_model}\
            --data_path ft-training_set/MetaMathQA-40K.json \
            --data_length 10000000 \
            --bf16 True \
            --output_dir ./trained_models/{model}_metamath_epoch{epoch}_r{r}_scale{scale}_lr{lr}_ratio{ratio}_seed{seed}/\
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps {bs//4} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate {lr}\
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --logging_steps 50 \
            --num_train_epochs {epoch} \
            --target_modules q_proj k_proj v_proj up_proj down_proj \
            --lora_r {r}\
            --lora_alpha {alpha}\
            --seed {seed}\
            --lora_dropout 0.05\
            --lora_init {init}\
            --prepare_ratio {ratio}\
                ')
input()