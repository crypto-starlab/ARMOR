import math
from collections import OrderedDict

import torch


def binomial_threshold(dim, p_value):

    k = dim // 2
    pmf = math.exp(math.lgamma(dim + 1) - math.lgamma(k + 1)
                   - math.lgamma(dim - k + 1) - dim * math.log(2))
    cdf = 0.5 + pmf / 2 if dim % 2 == 0 else 0.5
    while cdf < 1 - p_value and k < dim:
        pmf *= (dim - k) / (k + 1)
        k += 1
        cdf += pmf
    return k + 1


class PoisonedFL:
    def __init__(self, model, args):
        self.names = [name for name, p in model.named_parameters() if p.requires_grad]
        if not self.names:
            raise ValueError('PoisonedFL requires trainable parameters')
        self.scale = float(args.poisonedfl_scale)
        self.window = int(args.poisonedfl_window)
        self.decay = float(args.poisonedfl_decay)
        self.min_scale = float(args.poisonedfl_min_scale)
        p_value = float(args.poisonedfl_pvalue)
        if not (math.isfinite(self.scale) and math.isfinite(self.min_scale)
                and 0 < self.min_scale <= self.scale):
            raise ValueError('PoisonedFL requires finite scale >= min_scale > 0')
        if self.window < 1 or not 0 < self.decay < 1 or not 0 < p_value < 0.5:
            raise ValueError('Invalid PoisonedFL window, decay, or pvalue')
        current = self._vector(model.state_dict())
        generator = torch.Generator(device='cpu').manual_seed(args.seed)
        self.sign = torch.randint(0, 2, current.shape, generator=generator).float() * 2 - 1
        self.threshold = binomial_threshold(current.numel(), p_value)
        self.previous_model = None
        self.previous_attack = None
        self.window_model = None
        self.feedback_rounds = 0
        self.last_stats = None

    def _vector(self, state):
        return torch.cat([state[name].detach().to(device='cpu', dtype=torch.float32).flatten()
                          for name in self.names])

    @torch.no_grad()
    def craft(self, model):
        state = model.state_dict()
        current = self._vector(state)
  
        if self.previous_model is None:
            attack = torch.zeros_like(current)
            self.window_model = current.clone()
            history_norm = 0.0
            aligned = None
        else:
            history = current - self.previous_model
            history_norm = torch.linalg.vector_norm(history)
            self.feedback_rounds += 1
            aligned = None
            if self.feedback_rounds % self.window == 0:
                total = current - self.window_model
                total = torch.where(total == 0, current, total)
                aligned = int((total.sign() == self.sign).sum())
                if aligned < self.threshold and self.scale * self.decay >= self.min_scale:
                    self.scale *= self.decay
                self.window_model = current.clone()
            previous_norm = torch.linalg.vector_norm(self.previous_attack)
            residual = (history - self.previous_attack * history_norm / (previous_norm + 1e-9)).abs()
            direction = residual / (torch.linalg.vector_norm(residual) + 1e-9)
            attack = self.scale * history_norm * direction * self.sign
            history_norm = float(history_norm)
        self.previous_model = current
        self.previous_attack = attack.clone()
        self.last_stats = dict(scale=self.scale, history_norm=history_norm,
                               attack_norm=float(torch.linalg.vector_norm(attack)),
                               aligned=aligned, threshold=self.threshold)
        result = OrderedDict((name, torch.zeros_like(value)) for name, value in state.items())
        offset = 0
        for name in self.names:
            value = state[name]
            result[name] = attack[offset:offset + value.numel()].reshape(value.shape).to(value)
            offset += value.numel()
        return result
