from collections import OrderedDict
import math
from statistics import NormalDist

import torch


ALIASES = {
    'alie': 'alie', 'alie_attack': 'alie', 'alieattack': 'alie',
    'minsum': 'min_sum', 'min_sum': 'min_sum', 'min_sum_attack': 'min_sum',
    'minmax': 'min_max', 'min_max': 'min_max', 'min_max_attack': 'min_max',
    'sign_flipping': 'sign_flipping', 'signflipping': 'sign_flipping',
    'sign_flip': 'sign_flipping',
}


def attack_name(name):
    return ALIASES.get(str(name).lower().replace('-', '_').replace(' ', '_'))


def craft_attack(updates, parameter_names, name, args, num_clients, num_malicious):

    if not updates:
        raise ValueError('Attack requires at least one reference update')
    keys = [k for k in updates[0] if k in parameter_names]
    if not keys:
        raise ValueError('Attack requires trainable parameters')
    means, stds = {}, {}
    device = updates[0][keys[0]].device
    n = len(updates)
    pairwise = torch.zeros((n, n), dtype=torch.float64, device=device)
    for key in keys:
        values = torch.stack([u[key].detach().reshape(-1).float() for u in updates])
        means[key] = values.mean(0)
        stds[key] = values.std(0, unbiased=False)
        if name in {'min_sum', 'min_max'}:
            # Explicit differences avoid cancellation for nearly equal updates.
            for i in range(n):
                pairwise[i] += (values - values[i]).double().square().sum(1)

    if name == 'alie':
        z = getattr(args, 'alie_z', None)
        if z is None:
            s = math.floor(num_clients / 2 + 1) - num_malicious
            probability = (num_clients - num_malicious - s) / (num_clients - num_malicious)
            z = max(0., NormalDist().inv_cdf(min(1 - 1e-12, max(1e-12, probability))))
        if not math.isfinite(z) or z < 0:
            raise ValueError('alie_z must be finite and nonnegative')
        attacked = {k: means[k] + z * stds[k] for k in keys}
        stats = {'z': z}
    else:
        direction_kind = getattr(args, 'min_direction', 'std')
        directions = {k: (stds[k] if direction_kind == 'std' else means[k].sign()
                          if direction_kind == 'sign' else means[k]) for k in keys}
        if direction_kind == 'unit':
            norm = sum(v.double().square().sum() for v in directions.values()).sqrt()
            directions = {k: v / norm.to(v) if norm > 0 else torch.zeros_like(v)
                          for k, v in directions.items()}
        a = torch.zeros((), dtype=torch.float64, device=device)
        b = torch.zeros(n, dtype=torch.float64, device=device)
        c = torch.zeros_like(b)
        for key in keys:
            d = directions[key].double()
            a += d.square().sum()
            for i, update in enumerate(updates):
                delta = means[key].double() - update[key].reshape(-1).double()
                b[i] += (delta * d).sum()
                c[i] += delta.square().sum()
        bound = pairwise.max() if name == 'min_max' else pairwise.sum(1).min()
        def feasible(value):
            distances = (a * value * value - 2 * b * value + c).clamp_min(0)
            score = distances.max() if name == 'min_max' else distances.sum()
            return bool(score <= bound)
        cap = getattr(args, 'min_lambda_max', 10.)
        tolerance = getattr(args, 'min_tolerance', 1e-5)
        if not math.isfinite(cap) or cap <= 0 or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('min_lambda_max and min_tolerance must be finite and positive')
        low, high = 0., cap
        if a > 0:
            for _ in range(80):
                if high - low <= tolerance:
                    break
                mid = (low + high) / 2
                if feasible(mid):
                    low = mid
                else:
                    high = mid
            if feasible(cap):
                low = cap
        attacked = {k: means[k] - low * directions[k] for k in keys}
        stats = {'lambda': low, 'bound': float(bound)}
    result = OrderedDict((k, attacked[k].reshape(v.shape).to(v).clone()
                          if k in attacked else torch.zeros_like(v))
                         for k, v in updates[0].items())
    return result, stats


def apply_attacks(updates, positions, model, args):
    name = attack_name(args.attack)
    if not positions or name is None:
        return {}
    parameters = set(dict(model.named_parameters()))
    if name == 'sign_flipping':
        scale = getattr(args, 'sign_flip_scale', 1.)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('sign_flip_scale must be finite and nonnegative')
        for i in positions:
            updates[i] = OrderedDict((k, -scale * v if k in parameters else torch.zeros_like(v))
                                     for k, v in updates[i].items())
        return {'scale': scale}
    source = getattr(args, 'attack_reference', 'malicious')
    refs = ([u for i, u in enumerate(updates) if i not in positions]
            if source == 'benign' else [updates[i] for i in positions])
    if len(positions) == len(updates) and name == 'alie' and getattr(args, 'alie_z', None) is None:
        raise ValueError('Automatic ALIE z requires at least one benign client; specify --alie_z')
    crafted, stats = craft_attack(refs, parameters, name, args, len(updates), len(positions))
    for i in positions:
        updates[i] = OrderedDict((k, v.clone()) for k, v in crafted.items())
    return stats
