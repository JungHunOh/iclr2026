import os

print('enter gpu')
gpu=int(input())

dataset='commonsense_170k'

#model='gpt'
#model='llama'
model='llama3'

lr=2e-4
epoch=3
bs=32

if model == 'gpt':
    base_model = 'EleutherAI/gpt-j-6b'
    target_modules = ["q_proj", "k_proj", "v_proj", "fc_in", "fc_out"]
elif model == 'llama':
    base_model = 'yahma/llama-7b-hf'
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
elif model == 'llama3':
    base_model = 'meta-llama/Meta-Llama-3-8B'
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]

for seed in [1]:
    for r, scale, ratio in [(16,8,0.25)]:
        #os.system(f'CUDA_VISIBLE_DEVICES={gpu} python finetune.py --base_model {base_model} --data_path ./ft-training_set/{dataset}.json --output_dir ./trained_models/{model}_{dataset}_epoch{epoch}_lora_r{r}_scale{scale}_lr{lr}_seed{seed}/ --batch_size {bs} --micro_batch_size 16 --num_epochs {epoch}   --learning_rate {lr}   --cutoff_len 256   --val_set_size 0  --adapter_name lora --lora_r {r} --lora_alpha {scale * (r**0.5)}')
        os.system(f'CUDA_VISIBLE_DEVICES={gpu} python finetune.py --base_model {base_model} --data_path ./ft-training_set/{dataset}.json --output_dir ./trained_models/{model}_{dataset}_epoch{epoch}_lora_r{r}_scale{scale}_lr{lr}_seed{seed}/ --batch_size {bs} --micro_batch_size 16 --num_epochs {epoch}   --learning_rate {lr}   --cutoff_len 256   --val_set_size 0  --adapter_name lora --lora_r {r} --lora_alpha {scale * (r**0.5)} --prepare_ratio {ratio} --target_modules \"{','.join(target_modules)}\"')

        if dataset == 'commonsense_170k':
            #evalsets = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"]
            #evalsets = "boolq,piqa,social_i_qa,winogrande,ARC-Challenge,ARC-Easy,openbookqa"
            evalsets = "ARC-Challenge"
            eval_file = 'commonsense_evaluate.py'
        else:
            evalsets = 'SVAMP,AQuA,AddSub,gsm8k,MultiArith,SingleEq'
            eval_file = 'evaluate.py'
        
        if model == 'gpt':
            model_name = 'GPT-j-6B'
        elif model == 'llama':
            model_name = 'LLaMA-7B'
        elif model == 'llama3':
            model_name = 'LLaMA3-8B'
        #os.system(f'CUDA_VISIBLE_DEVICES={gpu} python {eval_file} --model {model_name} --adapter LoRA --datasets {evalsets} --base_model {base_model} --lora_weights ./trained_models/{model}_{dataset}_dl{dl}bs{bs}epoch{epoch}_lora_r{r}_target_r{target_r}_lr{lr}_seed{seed}')

