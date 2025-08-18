import os

print('PID:', os.getpid())

print('enter gpu')
gpu=input()

#model='gpt'
#model='llama'
#model='llama2'
model='llama3'

if model == 'gpt':
    base_model = 'EleutherAI/gpt-j-6b'
elif model == 'llama':
    base_model = 'yahma/llama-7b-hf'
elif model == 'llama2':
    base_model = "meta-llama/Llama-2-7b-hf"
elif model == 'llama3':
    base_model = 'meta-llama/Meta-Llama-3-8B'

bs=32

ratio=0
init='True'
max_steps=-1
epoch=3
for seed in [1]:
    for lr in [1e-3,5e-4]:
        for r, scale in [(32,1),(32,2)]:
            for method in ['base']:
                alpha = int(scale * (r**0.5))
                os.system(f'CUDA_VISIBLE_DEVICES={gpu} python train_math.py \
                    --model_name_or_path {base_model}\
                    --data_path ft-training_set/MetaMathQA-40K.json \
                    --data_length 10000000 \
                    --bf16 True \
                    --output_dir ./trained_models/{model}_metamath_epoch{epoch}_bs{bs}_r{r}_scale{scale}_lr{lr}_{method}_seed{seed}/\
                    --per_device_train_batch_size 8 \
                    --per_device_eval_batch_size 4 \
                    --gradient_accumulation_steps {bs//8} \
                    --evaluation_strategy "no" \
                    --save_strategy "no" \
                    --learning_rate {lr}\
                    --weight_decay 0. \
                    --warmup_ratio 0 \
                    --logging_steps 20 \
                    --max_steps {max_steps} \
                    --num_train_epochs {epoch} \
                    --lr_scheduler_type "cosine" \
                    --target_modules q_proj k_proj v_proj \
                    --lora_r {r}\
                    --lora_alpha {alpha}\
                    --seed {seed}\
                    --lora_dropout 0.05\
                    --lora_init {init}\
                    --prepare_ratio {ratio}\
                        ')