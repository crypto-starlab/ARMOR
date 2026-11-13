"""Run MNIST, FMNIST, CIFAR10 sequentially on GPU 0 over emulated LAN."""
import datetime
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

from run_client_results import ROLES, stop, summarize

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['MNIST', 'FMNIST', 'CIFAR10'],
                        default=['MNIST', 'FMNIST', 'CIFAR10'])
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--attack-start-round', type=int, default=50,
                        help='First attacked round, counted from 1')
    args = parser.parse_args()
    if not 1 <= args.attack_start_round <= args.rounds:
        parser.error('Require 1 <= attack-start-round <= rounds')
    folder = ROOT / 'BreakDown' / ('MPAF_LAN_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True)
    print(f'Results: {folder}', flush=True)
    status = {'status': 'preflight', 'datasets': args.datasets,
              'gpu': 0, 'num_clients': 30, 'rounds': args.rounds, 'attack': 'MPAF',
              'attack_ratio': 0.3, 'attack_start_round': args.attack_start_round,
              'attack_time_zero_based': args.attack_start_round - 1,
              'added_rtt_ms': 0.2, 'rate_mbps_per_direction': 3000}

    def save():
        temp = folder / 'status.tmp'
        temp.write_text(json.dumps(status, indent=2) + '\n')
        temp.replace(folder / 'status.json')

    save()
    processes = []
    streams = []
    try:
        subprocess.run(['sudo', '-n', 'true'], check=True)
        subprocess.run([sys.executable, '-c',
                        'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'], check=True)
        namespaces = subprocess.check_output(['sudo', '-n', 'ip', 'netns', 'list'], text=True)
        for link, ns in [('server2', 'armor-s2'), ('client', 'armor-client')]:
            action = 'profile' if any(x.split()[0] == ns for x in namespaces.splitlines() if x.split()) else 'setup'
            subprocess.run(['sudo', '-n', 'bash', str(ROOT / 'network.sh'), '--link', link, action, 'lan'], check=True)
        env = dict(os.environ, ARMOR_PYTHON=sys.executable, ARMOR_AUTHKEY=secrets.token_hex(32),
                   NUM_CLIENTS='30', GLOBAL_EP=str(args.rounds), LOCAL_EP='1', LOCAL_BS='32',
                   ATTACK='MPAF', ATTACK_RATIO='0.3', ATTACK_TIME=str(args.attack_start_round - 1), SEED='2023',
                   KEYCENTER_GPU='0', SERVER1_GPU='0', SERVER2_GPU='0', CLIENT_GPU='0',
                   CONNECT_TIMEOUT='600', PYTHONUNBUFFERED='1')
        for dataset in status['datasets']:
            output = ROOT / 'BreakDown' / dataset / folder.name
            output.mkdir(parents=True)
            env.update(DATASET=dataset, RUN_TAG=f'{folder.name}_{dataset}')
            status.update(status='running', current_dataset=dataset)
            save()
            with (output / 'network_status.log').open('w') as log:
                subprocess.run(['sudo', '-n', 'bash', str(ROOT / 'network.sh'), 'status'],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            processes, streams = [], []
            try:
                for role in ROLES:
                    command = ['bash', str(ROOT / 'lan.sh'), role]
                    if role == 'keycenter':
                        command += ['--log_path', str(output / (dataset.lower() + '_{role}.jsonl'))]
                    stream = (output / f'{role}.stdout.log').open('w')
                    streams.append(stream)
                    processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                                      stderr=subprocess.STDOUT, process_group=0))
                deadline = time.monotonic() + 48 * 3600
                renewed = time.monotonic()
                while any(p.poll() is None for p in processes):
                    bad = [(r, p.returncode) for r, p in zip(ROLES, processes) if p.poll() not in (None, 0)]
                    if bad:
                        raise RuntimeError(f'{dataset}: role failure {bad}')
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'{dataset}: exceeded 48 hours')
                    if time.monotonic() - renewed > 60:
                        subprocess.run(['sudo', '-n', '-v'], check=True)
                        renewed = time.monotonic()
                    time.sleep(2)
                if any(p.returncode != 0 for p in processes):
                    raise RuntimeError(f'{dataset}: role failure')
                if not summarize(output, 30, args.rounds, dataset=dataset):
                    raise RuntimeError(f'{dataset}: incomplete rounds')
                status.setdefault('completed_datasets', []).append(dataset)
                save()
                print(f'Completed: {dataset}', flush=True)
            finally:
                stop(processes)
                for stream in streams:
                    stream.close()
        status['status'] = 'completed'
        save()
    except BaseException as exc:
        status.update(status='failed', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
