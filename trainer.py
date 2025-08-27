from transformers import Trainer
import torch
import math
import random
from torch.utils.data import SequentialSampler

class CustomLoRATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _get_train_sampler(self, train_dataset=None):
        if self.args.max_steps > 0 or not hasattr(self, 'do_random_sampler'):
            if train_dataset is None:
                train_dataset = self.train_dataset
            self.do_random_sampler = True
            return SequentialSampler(self.train_dataset)
        else:
            return super()._get_train_sampler(train_dataset)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        if self.optimizer is None:
            self.create_optimizer()
        
        param_groups = self.optimizer.param_groups
        assert len(param_groups) > 0

        # Copy all param_groups to the new optimizer
        new_param_groups = []
        for group in param_groups:
            group_copy = {
                'params': group['params'],
                'lr': group.get('lr', 1e-3),
                'betas': group.get('betas', (0.9, 0.999)),
                'eps': group.get('eps', 1e-8),
                'weight_decay': group.get('weight_decay', 0.0),
                'amsgrad': getattr(self.optimizer, 'amsgrad', False)
            }
            new_param_groups.append(group_copy)
        
        mode = self.args.output_dir.split('_')[-2]
        assert type(self.optimizer) is torch.optim.AdamW, "only support AdamW optimizer"
        self.optimizer = CustomAdamW(
            new_param_groups,
            model=self.model,
            mode=mode,
            before_init=self.args.max_steps > 0
        )

        if self.args.max_steps > 0:
            self.lr_scheduler = NoScheduling(self.optimizer)
        else:
            self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

class CustomAdamW(torch.optim.AdamW):
    def __init__(self, params, model=None, mode='base', before_init=False, **kwargs):
        super().__init__(params, **kwargs)
        self.model = model
        self._step_count = 0
        self.mode = mode
        self.before_init = before_init
        self.interval = int(self.mode.split('interval')[-1]) if 'interval' in self.mode else 1

        if hasattr(self.model, 'classifier'):
            self.classifier_params = [p for p in self.model.classifier.parameters() if p.requires_grad]
        elif hasattr(self.model, 'classification_head'):
            self.classifier_params = [p for p in self.model.classification_head.parameters() if p.requires_grad]

        layer = 0
        for module in self.model.modules():
            if hasattr(module, 'lora_A'):
                if not hasattr(module, 'layer_idx'):
                    module.layer_idx = layer
                    layer += 1
                if not self.before_init and 'init' in self.mode:
                    if self.mode == 'oursnewinitnoproj':
                        module.lora_B['default'].weight.data = module.detached_b.clone().contiguous()
                        module.lora_A['default'].weight.data = module.detached_a.clone().contiguous()
                        del module.prev_a
                        del module.prev_b
                    elif self.mode == 'oursinitone':
                        module.do_one = True
                        module.proj_a = module.detached_a.clone().contiguous()
                        module.proj_b = module.detached_b.clone().contiguous()
                        module.lora_B['default'].weight.data = module.detached_b.clone().contiguous()
                        module.lora_A['default'].weight.data = module.detached_a.clone().contiguous()
                        module.base_layer.weight.data = module.base_layer.weight - (module.detached_b @ module.detached_a).to(module.base_layer.weight.dtype) * module.scaling['default'] * 2
                        del module.prev_a
                        del module.prev_b
                        del module.detached_a
                        del module.detached_b
                    else:
                        module.proj_a = module.detached_a.clone().contiguous()
                        module.proj_b = module.detached_b.clone().contiguous()
                        module.lora_B['default'].weight.data = module.detached_b.clone().contiguous()
                        module.lora_A['default'].weight.data = module.detached_a.clone().contiguous()
                        #torch.nn.init.zeros_(module.lora_A['default'].weight)
                        #torch.nn.init.zeros_(module.lora_B['default'].weight)
                        module.base_layer.weight.data = module.base_layer.weight - (module.detached_b @ module.detached_a).to(module.base_layer.weight.dtype) * module.scaling['default']
                        del module.prev_a
                        del module.prev_b
                        del module.detached_a
                        del module.detached_b

    def step(self, closure=None):
        loss = super().step(closure)

        self._step_count += 1

        # if self._step_count % 10 == 0 and not self.before_init:
        #     for module in self.model.modules():
        #         if hasattr(module, 'lora_A'):
        #             with torch.no_grad():
        #                 if module.layer_idx % 2 == 0:
        #                     lora_A = module.lora_A['default'].weight
        #                     lora_B = module.lora_B['default'].weight
        #                     u, s, v = torch.svd_lowrank(lora_B @ lora_A, q=module.lora_A['default'].weight.shape[0], niter=4)
        #                     rank1 = (u[:,0:1] @ v[:,0:1].T).reshape(-1)
        #                     cos = []
        #                     for j in range(lora_A.shape[0]):
        #                         ba = (lora_B[:,j:j+1] @ lora_A[j:j+1,:]).reshape(-1)
        #                         cos.append(torch.dot(ba, rank1) / (torch.norm(ba) * torch.norm(rank1)))
        #                     print(module.layer_idx,'cosine similarity:', torch.mean(torch.tensor(cos)).item())

        if self._step_count % 10 == 0 and self.before_init:
            for module in self.model.modules():
                if hasattr(module, 'lora_A'):
                    with torch.no_grad():
                        lora_A = module.lora_A['default'].weight
                        lora_B = module.lora_B['default'].weight
                        if hasattr(module, 'prev_a') and hasattr(module, 'prev_b'):
                            u, s, v = torch.svd_lowrank(lora_B @ lora_A - module.prev_b @ module.prev_a, q=module.lora_A['default'].weight.shape[0], niter=4)
                        else:
                            u, s, v = torch.svd_lowrank(lora_B @ lora_A, q=module.lora_A['default'].weight.shape[0], niter=4)
                        module.lora_B['default'].weight.data = u.clone().contiguous()
                        module.lora_A['default'].weight.data = v.T.clone().contiguous()
                        module.detached_b = u.clone().contiguous()
                        module.detached_a = v.T.clone().contiguous()
                        module.prev_b = u.clone().contiguous()
                        module.prev_a = v.T.clone().contiguous()
                        if hasattr(self.model, 'classifier'):
                            for p, p_init in zip(self.model.classifier.parameters(), self.classifier_params):
                                if p.requires_grad:
                                    p.data = p_init.data.clone().contiguous()
                        elif hasattr(self.model, 'classification_head'):
                            for p, p_init in zip(self.model.classification_head.parameters(), self.classifier_params):
                                if p.requires_grad:
                                    p.data = p_init.data.clone().contiguous()
                        self.state.clear()

        if not self.before_init and 'ours' in self.mode and self._step_count % self.interval == 0:
            for module in self.model.modules():
                if hasattr(module, 'lora_A'):
                    with torch.no_grad():
                        lora_A = module.lora_A['default'].weight
                        lora_B = module.lora_B['default'].weight
                        if (torch.norm(module.lora_B['default'].weight, dim=0) > 1e-2).all() and (torch.norm(module.lora_A['default'].weight, dim=1) > 1e-2).all():
                            Q_A, R_A = torch.linalg.qr(lora_A.T, mode='reduced')
                            Q_B, R_B = torch.linalg.qr(lora_B, mode='reduced')
                            if 'qrab' in self.mode:
                                module.lora_A['default'].weight.data = (Q_A * torch.diag(R_A)).T.clone().contiguous()
                                module.lora_B['default'].weight.data = (Q_B * torch.diag(R_B)).clone().contiguous()
                            if 'noproj' not in self.mode:
                                if 'scaling' in self.mode:
                                    module.proj_a = (Q_A * torch.diag(R_A)).T.clone().contiguous()
                                    module.proj_b = (Q_B * torch.diag(R_B)).clone().contiguous()
                                else:
                                    module.proj_a = (Q_A * torch.sign(torch.diag(R_A))).T.clone().contiguous()
                                    module.proj_b = (Q_B * torch.sign(torch.diag(R_B))).clone().contiguous()
                            
        return loss

import torch
from torch.optim.lr_scheduler import _LRScheduler

class FixedThenLinearDecayWithWarmupLR(_LRScheduler):
    def __init__(self, optimizer, fixed_steps, total_steps, warmup_steps=0, final_lr=1e-6, last_epoch=-1):
        self.fixed_steps = fixed_steps
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.fixed_steps:
                # Fixed LR
                lr = base_lr
            elif step < self.fixed_steps + self.warmup_steps:
                # Linear warmup
                # Linear warmup from 0 to base_lr
                progress = (step - self.fixed_steps) / max(1, self.warmup_steps)
                lr = base_lr * progress
            else:
                # Linear decay after fixed_steps
                decay_steps = self.total_steps - self.fixed_steps
                t = min(max(0, step - self.fixed_steps), decay_steps) / max(1, decay_steps)
                lr = self.final_lr + (base_lr - self.final_lr) * (1 - t)
            lrs.append(lr)
        return lrs

class NoScheduling(_LRScheduler):
    def __init__(self, optimizer=None, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [lr for lr in self.base_lrs]
