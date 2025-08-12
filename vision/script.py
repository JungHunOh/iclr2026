import os
import math

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

cuda_visible_devices = input()  # Set your desired GPU index here

ii = int(cuda_visible_devices)

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

for dataset in ['resisc45']:
#for dataset in ['sun397']:
    bs = 128
    model='dino-v2-base'
    lr = 1e-3

    method='lora'
    #method='oh'

    init='True'
    #init='pissa_niter_4'

    if dataset in ['resisc45', 'sun397']:
        epoch = 10
    elif dataset in ['cifar100', 'food101']:
        epoch = 5
    elif dataset in ['dtd', 'cars', 'cub200']:
        epoch = 20

    ratio = 0

    for seed in [1,2,3]:
        for mode in ['oursnew']:
            for r in [32]:
                for j in [ii]:
                    scale = round(0.5 + j * 0.5,1)
                    
                    alpha = math.sqrt(r) * scale
                    if method == 'lora':
                        output_dir = f"./experiment/{dataset}/{model}_lora_epoch{epoch}_bs{bs}_init{init}_lr{lr}_alpha{scale}_r{r}_{mode}/"
                    elif method == 'oh':
                        output_dir = f"./experiment/{dataset}/{model}_lora_epoch{epoch}_bs{bs}_init{init}_lr{lr}_alpha{scale}_r{r}_{mode}/"
                    cmd = (
                        f"python finetune.py "
                        f"--eval_strategy no "
                        f"--save_strategy no "
                        f"--gradient_accumulation_steps {bs//64} "
                        f"--dataloader_num_workers 8 "
                        f"--logging_steps 100 "
                        f"--label_names labels "
                        f"--remove_unused_columns False "
                        f"--per_device_train_batch_size 64 "
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
                        f"--warmup_ratio 0 "
                        f"--weight_decay 0.01 "
                        f"--lora_alpha {alpha} "
                        f"--lora_init {init} "
                        f"--prepare_ratio {ratio} "
                    )
                    os.system(cmd)
