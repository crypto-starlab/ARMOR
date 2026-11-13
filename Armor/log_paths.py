"""Canonical destinations for accuracy logs."""
from pathlib import Path

ACCURACY_ROOT = Path(__file__).resolve().parent / 'accuracy_logs'
MAIN_ATTACKS = {'poisonedfl', 'mpaf', 'scaling'}


def normalize_attack(name):
    name = str(name).lower().replace('-', '_')
    aliases = {'poisoned_fl': 'poisonedfl', 'min_sum': 'minsum', 'min_max': 'minmax',
               'sign_flip': 'sign_flipping', 'signflipping': 'sign_flipping',
               'label_flip': 'label_flipping', 'labelflipping': 'label_flipping',
               'alie_attack': 'alie'}
    return aliases.get(name, name)


def accuracy_log_path(dataset, method, attack, ratio=0.3, root=None):
    dataset = str(dataset).upper()
    attack = normalize_attack(attack)
    attack_label = 'MPAF' if attack == 'mpaf' else attack
    base = ACCURACY_ROOT if root is None else Path(root)
    folder = base / dataset if attack in MAIN_ATTACKS else base / 'SupplyMentary' / dataset
    if method == 'armor' and attack in MAIN_ATTACKS and float(ratio) in {0.1, 0.2, 0.4}:
        folder = base / 'SupplyMentary' / 'Attack_Ratios' / dataset / f'{ratio:g}'
    return folder / f'{dataset}_{method}_{attack_label}_{ratio:g}_log.txt'
