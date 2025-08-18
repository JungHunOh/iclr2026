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
        
        #mode = self.args.output_dir.split('_')[-1].replace('/','')
        mode = self.args.output_dir.split('_')[-2]
        for module in self.model.modules():
            if hasattr(module, 'lora_A'):
                module.method = mode

        assert type(self.optimizer) is torch.optim.AdamW, "only support AdamW optimizer"
        self.optimizer = CustomAdamW(
            new_param_groups,
            target_iter=int(self.prepare_ratio * num_training_steps),
            model=self.model,
            mode=mode,
            before_init=self.args.max_steps > 0
        )

        if self.prepare_ratio > 0:
            self.lr_scheduler = FixedThenLinearDecayWithWarmupLR(self.optimizer, fixed_steps=int(num_training_steps*self.prepare_ratio), total_steps=num_training_steps, warmup_steps=self.args.warmup_ratio * num_training_steps * (1-self.prepare_ratio))
        else:
            self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

class CustomAdamW(torch.optim.AdamW):
    def __init__(self, params, target_iter=0, model=None, mode='base', before_init=False, **kwargs):
        super().__init__(params, **kwargs)
        self.model = model
        self._step_count = 0
        self.target_iter = target_iter
        self.mode = mode
        self.before_init = before_init
        if 'svdevery' in self.mode:
            self.interval = int(self.mode.split('svdevery')[-1])
        if hasattr(self.model, 'classifier'):
            try:
                self.classifier_init = self.model.classifier.weight.clone()
            except:
                pass

        layer = 0
        for module in self.model.modules():
            if hasattr(module, 'lora_A'):
                module.layer_idx=layer
                layer += 1
                module.iter = 0
                module.first_micro_batch = True
                module.rank = module.lora_A['default'].weight.shape[0]

    def step(self, closure=None):
        loss = super().step(closure)

        self._step_count += 1

        ranks = []
        for module in self.model.modules():
            if hasattr(module, 'lora_A'):
                module.iter += 1
                module.first_micro_batch = True
                scale = 1
                if self._step_count > self.target_iter and 'oursnew' in self.mode and (torch.norm(module.lora_B['default'].weight[:,:module.rank], dim=0) > 1e-2).all() and (torch.norm(module.lora_A['default'].weight[:module.rank], dim=1) > 1e-2).all():
                    with torch.no_grad():
                        W = module.base_layer.weight
                        scaling = module.scaling['default']
                        lora_A = module.lora_A['default'].weight[:module.rank]
                        lora_B = module.lora_B['default'].weight[:,:module.rank]
                        #module.base_layer.weight.data = W + (lora_B @ lora_A).clone().contiguous().to(W.dtype) * scaling
                        Q_A, R_A = torch.linalg.qr(lora_A.T, mode='reduced')
                        Q_B, R_B = torch.linalg.qr(lora_B, mode='reduced')
                        #if self._step_count % 20 == 0 and module.layer_idx % 7 == 0:
                        if False:
                            if hasattr(module, 'prev_b') and hasattr(module, 'prev_a'):
                                ba = (lora_B @ lora_A).reshape(-1)
                                prev_ba = torch.nn.functional.normalize((module.prev_b @ module.prev_a).reshape(-1),dim=0)
                                print((torch.nn.functional.normalize(ba,dim=0)*prev_ba).sum(),module.layer_idx)
                            module.prev_b = module.lora_B['default'].weight.clone().contiguous()
                            module.prev_a = module.lora_A['default'].weight.clone().contiguous()
                            
                        module.lora_A['default'].weight.data[:module.rank] = (Q_A * (torch.diag(R_A)) * scale).T.clone().contiguous()
                        module.lora_B['default'].weight.data[:,:module.rank] = (Q_B *(torch.diag(R_B)) * scale).clone().contiguous()
                        #module.base_layer.weight.data = W - (module.lora_B['default'].weight @ module.lora_A['default'].weight).clone().contiguous().to(W.dtype) * scaling
                if 'init' in self.mode and self._step_count % 10 == 0 and self._step_count < self.target_iter:
                    if module.layer_idx == 1:
                        if len(self.param_groups[0]['params']) > len(self.param_groups[1]['params']):
                            idx = 0
                        else:
                            idx = 1
                        for p in self.param_groups[idx]['params']:
                            if p in self.state:
                                state = self.state[p].clear()
                    with torch.no_grad():
                        scaling = module.scaling['default']
                        lora_A = module.lora_A['default'].weight[:module.rank]
                        lora_B = module.lora_B['default'].weight[:,:module.rank]
                        module.scaling2 = 1
                        if True:
                            if hasattr(module, 'prev_a') and hasattr(module, 'prev_b'):
                                u, s, v = torch.svd_lowrank(lora_B @ lora_A - module.prev_b @ module.prev_a, q=module.rank, niter=4)
                            else:
                                u, s, v = torch.svd_lowrank(lora_B @ lora_A, q=module.rank, niter=4)
                            module.lora_B['default'].weight.data[:,:module.rank] = u.clone().contiguous()
                            module.lora_A['default'].weight.data[:module.rank] = v.T.clone().contiguous()
                            module.detached_b = u.clone().contiguous()
                            module.detached_a = v.T.clone().contiguous()
                            module.prev_b = module.lora_B['default'].weight.clone().contiguous()
                            module.prev_a = module.lora_A['default'].weight.clone().contiguous()
                            if hasattr(self.model, 'classifier'):
                                self.model.classifier.weight.data = self.classifier_init
                                torch.nn.init.zeros_(self.model.classifier.bias)
                                self.state.clear()
                        elif False:
                            if module.layer_idx % 7 == 0:
                                prev_BA = torch.nn.functional.normalize((module.prev_b @ module.prev_a).reshape(-1),dim=0)
                                BA = torch.nn.functional.normalize((lora_B @ lora_A).reshape(-1),dim=0)
                                print((BA * prev_BA).sum(), torch.norm((module.detached_b@module.detached_a).reshape(-1),dim=0),module.layer_idx)

                            #norm_a = torch.norm(module.detached_a, dim=1, keepdim=True) * 0.5 / (module.rank**0.5)
                            #norm_b = torch.norm(module.detached_b, dim=0, keepdim=True) * 0.5 / (module.rank**0.5)
                            norm_a = 1
                            norm_b = 1
                            Q_A, R_A = torch.linalg.qr(lora_A.T, mode='reduced')
                            module.detached_a = module.detached_a + (lora_A - (Q_A * torch.sign(torch.diag(R_A))).T * module.scaling2 * norm_a).clone().contiguous()
                            module.lora_A['default'].weight.data[:module.rank] = (Q_A * torch.sign(torch.diag(R_A))).T.clone().contiguous() * module.scaling2 * norm_a
                            Q_B, R_B = torch.linalg.qr(lora_B, mode='reduced')
                            module.detached_b = module.detached_b + (lora_B - (Q_B * torch.sign(torch.diag(R_B))) * module.scaling2 * norm_b).clone().contiguous()
                            module.lora_B['default'].weight.data[:,:module.rank] = (Q_B * torch.sign(torch.diag(R_B))).clone().contiguous() * module.scaling2 * norm_b
                            module.prev_b = module.lora_B['default'].weight.clone().contiguous()
                            module.prev_a = module.lora_A['default'].weight.clone().contiguous()
                            
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
