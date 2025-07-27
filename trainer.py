from transformers import Trainer
import torch
import math
import random

class CustomLoRATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.prepare_ratio = kwargs.pop('prepare_ratio', 0)
        super().__init__(*args, **kwargs)

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
        
        assert type(self.optimizer) is torch.optim.AdamW, "only support AdamW optimizer"
        self.optimizer = CustomAdamW(
            new_param_groups,
            target_iter=int(self.prepare_ratio * num_training_steps),
            model=self.model
        )

        if self.prepare_ratio > 0:
            self.lr_scheduler = FixedThenLinearDecayWithWarmupLR(self.optimizer, fixed_steps=int(num_training_steps*self.prepare_ratio), total_steps=num_training_steps, warmup_steps=self.args.warmup_ratio * num_training_steps * (1-self.prepare_ratio))
        else:
            self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

class CustomAdamW(torch.optim.AdamW):
    def __init__(self, params, target_iter=0, model=None, **kwargs):
        super().__init__(params, **kwargs)
        self.model = model
        self._step_count = 0
        self.target_iter = target_iter
        if hasattr(self.model, 'classifier'):
            self.classifier_init = self.model.classifier.weight.clone()

        layer = 0
        for module in self.model.modules():
            if hasattr(module, 'lora_A'):
                module.layer_idx=layer
                layer += 1

    def step(self, closure=None):
        loss = super().step(closure)
        
        self._step_count += 1

        if self._step_count == self.target_iter and self.target_iter > 0:
            self.state.clear()
            if hasattr(self.model, 'classifier'):
                self.model.classifier.weight.data = self.classifier_init
                torch.nn.init.zeros_(self.model.classifier.bias)
            with torch.no_grad():
                for module in self.model.modules():
                    if hasattr(module, 'lora_A'):
                        W = module.base_layer.weight
                        lora_A = module.lora_A['default'].weight
                        lora_B = module.lora_B['default'].weight
                        scaling = module.scaling['default']
                        r = lora_A.shape[0]

                        target_r = r

                        u, s, v = torch.linalg.svd(lora_B @ lora_A, full_matrices=True)
                        u = u[:,:target_r]
                        v = v[:target_r,:]
                        s = s[:target_r]

                        module.lora_A['default'].weight.data = v.clone().contiguous()
                        module.lora_B['default'].weight.data = u.clone().contiguous()

                        module.lora_A_init = v.clone().contiguous()
                        module.lora_B_init = u.clone().contiguous()

                        #module.scaling['default'] *= r/target_r / (r**0.5) * (target_r**0.5)

                        module.base_layer.weight.data = W - (module.lora_B['default'].weight @ module.lora_A['default'].weight * module.scaling['default']).to(W.dtype)

                        #module.detach_lora = True

        #if self._step_count > self.target_iter and self.target_iter > 0:
        if False:
            for module in self.model.modules():
                if hasattr(module, 'lora_A'):
                    with torch.no_grad():
                        W = module.base_layer.weight
                        lora_A = module.lora_A['default'].weight
                        lora_B = module.lora_B['default'].weight
                        scaling = module.scaling['default']
                        r = lora_A.shape[0]

                        ba = lora_B @ lora_A
                        ba_init = module.lora_B_init @ module.lora_A_init
                        
                        # if module.layer_idx % 13 == 0:
                        #     u, s, v = torch.linalg.svd(ba-ba_init)
                        #     s_norm = s / s.sum()
                        #     entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-12))
                        #     effective_rank = torch.exp(entropy)
                        #     print(f"Effective rank: {effective_rank.item():.4f}")

                        module.base_layer.weight.data = W  + ((ba - ba_init)* scaling).to(W.dtype)

                        module.lora_A['default'].weight.data = module.lora_A_init.clone()
                        module.lora_B['default'].weight.data = module.lora_B_init.clone()

                        #module.lora_A_init = lora_A.clone()
                        #module.lora_B_init = lora_B.clone()



        #if self._step_count % 50 == 0 and self._step_count > self.target_iter:
        if False:
            for module in self.model.modules():
                if hasattr(module, 'lora_A'):
                    W = module.base_layer.weight
                    lora_A = module.lora_A['default'].weight
                    lora_B = module.lora_B['default'].weight
                    scaling = module.scaling['default']
                    r = lora_A.shape[0]

                    if module.layer_idx % 19 == 0:
                        u, s, v = torch.linalg.svd(lora_B @ lora_A)
                        s_norm = s / s.sum()
                        entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-12))
                        effective_rank = torch.exp(entropy)
                        print(f"Effective rank: {effective_rank.item():.4f}")

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
