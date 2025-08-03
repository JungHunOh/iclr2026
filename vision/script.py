import os
import math

cuda_visible_devices = input()  # Set your desired GPU index here

ii = int(cuda_visible_devices)

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

for dataset in ['resisc45', 'dtd', 'cars','cifar100', 'food101', 'flowers102', 'sun397']:
    bs = 64
    model='vit-base'
    lr = 4e-3

    method='lora'
    #method='oh'

    init='True'
    #init='pissa_niter_4'

    epoch = 10

    if method == 'lora':
        ratio = 0
    else:
        ratio = 1-10/epoch

    for seed in [1,2,3,4,5]:
        for mode in ['base','basesvd1','ours','ourssvd1','ourssvdr']:
        #for mode in ['base']:
            for r in [4,16,32]:
                for scale in [2+4*ii]:
                    alpha = int(math.sqrt(r) * scale)
                    if method == 'lora':
                        output_dir = f"./experiment/{dataset}/{model}_lora_epoch{epoch}_bs{bs}_init{init}_lr{lr}_alpha{scale}_r{r}_{mode}/"
                    elif method == 'oh':
                        output_dir = f"./experiment/{dataset}/{model}_lora_epoch{epoch}_bs{bs}_init{init}_lr{lr}_alpha{scale}_r{r}_{mode}/"
                    #output_dir = f"./cub200_results/test/"
                    cmd = (
                        f"python finetune.py "
                        f"--eval_strategy no "
                        f"--save_strategy no "
                        f"--gradient_accumulation_steps 1 "
                        f"--dataloader_num_workers 8 "
                        f"--logging_steps 10 "
                        f"--label_names labels "
                        f"--remove_unused_columns False "
                        f"--per_device_train_batch_size {bs} "
                        f"--per_device_eval_batch_size 256 "
                        f"--seed {seed} "
                        f"--num_train_epochs {epoch} "
                        f"--lora_rank {r} "
                        f"--output_dir {output_dir} "
                        f"--model_name {model} "
                        f"--finetuning_method {method} "
                        f"--dataset_name {dataset} "
                        f"--clf_learning_rate {lr} "
                        f"--other_learning_rate {lr} "
                        f"--warmup_ratio 0.1 "
                        f"--weight_decay 0.01 "
                        f"--lora_alpha {alpha} "
                        f"--lora_init {init} "
                        f"--prepare_ratio {ratio} "
                    )
                    os.system(cmd)
