import os
import math

#os.environ["NCCL_P2P_DISABLE"] = "1"
#os.environ["NCCL_IB_DISABLE"] = "1"

cuda_visible_devices = input()  # Set your desired GPU index here

ii = int(cuda_visible_devices)

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

for dataset in ['resisc45','dtd','cifar100','food101','sun397','cars']:
    bs = 128
    model='vit-base'
    lr = 1e-4

    method='lora'
    #method='oh'

    init='True'
    #init='pissa_niter_4'

    epoch = 10

    #ratio = 1-20/25
    ratio = 0

    for seed in [1,2,3]:
        for mode in ['base', 'ours', 'oursdetach']:
        #for mode in ['base']:
            for r in [4,32,64]:
                for j in [ii]:
                    if dataset == 'cifar100' or dataset == 'food101' or dataset == 'sun397':
                        scale = round(0.2 + j * 0.1,1)
                    elif dataset == 'resisc45' or dataset == 'cars':
                        scale = 2 + j * 2
                    elif dataset == 'dtd':
                        scale = 1 + j * 1
                    
                    alpha = math.sqrt(r) * scale
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
