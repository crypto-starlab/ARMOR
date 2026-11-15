#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python main.py --gpu 0 --method fedavg --global_ep 200 --num_clients 50 --attack_ratio 0.3  --attack poisonedfl --poisonedfl_scale 8 --dataset MNIST

python main.py --gpu 0 --method armor --cipher --global_ep 200 --num_clients 50 --attack_ratio 0.3  --attack poisonedfl  --poisonedfl_scale 8 --armor_filter intersection --dataset MNIST

python main.py --gpu 0 --method fedavg --global_ep 200 --num_clients 50 --attack_ratio 0.3 --attack MPAF --dataset MNIST

python main.py --gpu 0 --method armor --cipher --global_ep 200 --num_clients 50 --attack_ratio 0.3  --attack MPAF --armor_filter intersection --dataset MNIST

python main.py --gpu 0 --method fedavg --global_ep 200 --num_clients 50 --attack_ratio 0.3 --attack scaling --dataset MNIST

python main.py --gpu 0 --method armor --cipher --global_ep 200 --num_clients 50 --attack_ratio 0.3  --attack scaling --armor_filter intersection --dataset MNIST
