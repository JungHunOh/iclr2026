import os
import math

#os.environ["NCCL_P2P_DISABLE"] = "1"
#os.environ["NCCL_IB_DISABLE"] = "1"

cuda_visible_devices = input()  # Set your desired GPU index here

ii = int(cuda_visible_devices)

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

for model in ['vit-base']:
    for dataset in ['cars']:
    #for dataset in ['sun397']:
        bs = 128

        method='lora'
        #method='oh'

        init='True'
        #init='pissa_niter_4'

        if dataset in ['resisc45', 'food101']:
            epoch = 10
        elif dataset in ['sun397']:
            epoch = 15
        elif dataset in ['cifar100']:
            epoch = 7
        elif dataset in ['dtd', 'cub200']:
            epoch = 20
        elif dataset in ['cars']:
            epoch = 20

        ratio = 0

        for target_modules in ['value', 'query value', 'query key value']:
            target_modules_name = target_modules.replace(' ', '').replace('value','v').replace('query','q').replace('key','k')

            for seed in [1]:
                for mode in ['oursnew']:
                    for lr in [5e-3]:
                        for r in [32]:
                            for j in [0]:
                                if dataset in ['resisc45', 'dtd', 'cars']:
                                    scale = round(1 + j * 2,1)
                                else:
                                    scale = round(0.5 + j * 1,1)
                                
                                alpha = math.sqrt(r) * scale
                                if method == 'lora':
                                    output_dir = f"./experiment/{dataset}/{model}_lora_epoch{epoch}_bs{bs}_init{init}_lr{lr}_alpha{scale}_r{r}_{mode}_{target_modules_name}/"
                                elif method == 'oh':
                                    output_dir = f"./experiment/{dataset}/{model}_lora_epoch{epoch}_bs{bs}_init{init}_lr{lr}_alpha{scale}_r{r}_{mode}_{target_modules_name}/"
                                cmd = (
                                    f"python finetune.py "
                                    f"--eval_strategy no "
                                    f"--save_strategy no "
                                    f"--gradient_accumulation_steps {bs//64} "
                                    f"--dataloader_num_workers 8 "
                                    f"--logging_steps 50 "
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
                                    f"--clf_learning_rate 2e-3 "
                                    f"--other_learning_rate {lr} "
                                    f"--warmup_ratio 0 "
                                    f"--weight_decay 0 "
                                    f"--lora_alpha {alpha} "
                                    f"--lora_init {init} "
                                    f"--prepare_ratio {ratio} "
                                    f"--target_modules {target_modules} "
                                )
                                os.system(cmd)
