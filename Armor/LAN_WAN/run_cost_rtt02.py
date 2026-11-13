#!/usr/bin/env python3
"""ARMOR: 30 clients, one FedAvg warmup + three measured rounds per dataset."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
ROLES = ('keycenter', 'server2', 'server1', 'client')
ADDRESSES = dict(keycenter='10.204.84.1', server1='10.204.84.2',
                 server2='10.204.84.3', client='10.204.84.4')
BRIDGE = 'ar02br'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def network_commands():
    commands = [['ip', 'link', 'add', BRIDGE, 'type', 'bridge'],
                ['ip', 'link', 'set', BRIDGE, 'up']]
    for i, role in enumerate(ROLES):
        ns, host, peer = f'ar02-{role}', f'ar02-{i}h', f'ar02-{i}p'
        commands += [
            ['ip', 'netns', 'add', ns],
            ['ip', 'link', 'add', host, 'type', 'veth', 'peer', 'name', peer],
            ['ip', 'link', 'set', peer, 'netns', ns],
            ['ip', 'link', 'set', host, 'master', BRIDGE],
            ['ip', 'link', 'set', host, 'mtu', '1500', 'up'],
            ['ip', '-n', ns, 'link', 'set', 'lo', 'up'],
            ['ip', '-n', ns, 'addr', 'add', ADDRESSES[role] + '/24', 'dev', peer],
            ['ip', '-n', ns, 'link', 'set', peer, 'mtu', '1500', 'up'],
            ['ethtool', '-K', host, 'tso', 'off', 'gso', 'off', 'gro', 'off'],
            ['ip', 'netns', 'exec', ns, 'ethtool', '-K', peer,
             'tso', 'off', 'gso', 'off', 'gro', 'off'],
            ['tc', 'qdisc', 'add', 'dev', host, 'root', 'netem', 'limit', '10000',
             'delay', '100us', 'rate', '3000mbit'],
        ]
    return commands


def commands(args, folder, dataset):
    native = ROOT.parent / 'phantom-fhe/build/lib'
    result = []
    for role in ROLES:
        command = ['sudo', '-n', '--preserve-env=ARMOR_AUTHKEY', 'ip', 'netns', 'exec',
                   f'ar02-{role}', 'env', 'PYTHONUNBUFFERED=1', 'PYTHONDONTWRITEBYTECODE=1',
                   f'PYTHONPATH={native}:{ROOT.parent}',
                   f'LD_LIBRARY_PATH={native}' + (':' + os.environ['LD_LIBRARY_PATH']
                                                 if os.environ.get('LD_LIBRARY_PATH') else ''),
                   sys.executable, '-u', str(ROOT / f'{role}.py'), '--gpu', str(args.gpu),
                   '--keycenter_host', ADDRESSES['keycenter'], '--keycenter_port', '19500',
                   '--server1_host', ADDRESSES['server1'], '--server1_port', '19501',
                   '--server2_host', ADDRESSES['server2'], '--server2_port', '19502',
                   '--connect_timeout', '600']
        if role == 'keycenter':
            command += ['--dataset', dataset, '--method', 'armor', '--num_clients', '30',
                        '--global_ep', '4', '--local_ep', '1', '--local_bs', '32',
                        '--lr', '0.01', '--seed', '2023', '--attack', 'No_Attack',
                        '--attack_ratio', '0', '--extract_partial_constant',
                        '--log_path', str(folder / (dataset.lower() + '_{role}.jsonl'))]
        result.append((role, command))
    return result


def summarize(folder, dataset):
    logs = {role: [json.loads(s) for s in (folder / f'{dataset.lower()}_{role}.jsonl')
                  .read_text().splitlines() if s.strip()] for role in ROLES}
    for role in ('client', 'server1'):
        rounds = [e['round'] for e in logs[role] if e['event'] == 'round_summary']
        if sorted(rounds) != [0, 1, 2, 3]:
            raise ValueError(f'{dataset}/{role}: incomplete or duplicate rounds: {rounds}')
    rows = []
    for r in range(4):
        events = [e for es in logs.values() for e in es if e.get('round') == r]
        def seconds(role, names):
            return sum(e['latency_s'] for e in events if e['role'] == role and e['event'] in names)
        enc = [e for e in events if e['role'] == 'client' and e['event'] == 'encryption']
        if len(enc) != 30 or {e['client_id'] for e in enc} != set(range(30)):
            raise ValueError(f'{dataset}/{r}: missing client encryption')
        if r and not all(any(e['event'] == kind for e in events)
                         for kind in ('norm_he', 'similarity_he', 'aggregation_he')):
            raise ValueError(f'{dataset}/{r}: missing ARMOR phases')
        row = dict(dataset=dataset, round=r + 1, measured=r > 0,
                   Train=seconds('client', {'training'}), Enc=seconds('client', {'encryption'}),
                   training_postprocess_s=seconds('client', {'training_postprocess'}),
                   HE_comp=seconds('server1', {'norm_he', 'similarity_he', 'aggregation_he'}),
                   Dec=sum(seconds(role, {'decrypt_compute'}) for role in ('server1', 'server2')),
                   round_wall_client_s=seconds('client', {'round_summary'}),
                   round_wall_server_s=seconds('server1', {'round_summary'}))
        receives = {e['message_id']: e for e in events
                    if e['event'] == 'communication' and e['direction'] == 'receive'}
        row.update(Client_Side_all_messages=0., Server_Side_all_messages=0., Client_Side_model_upload=0.)
        for e in events:
            if e['event'] != 'communication' or e['direction'] != 'send':
                continue
            rx = receives[e['message_id']]
            if rx['frame_bits'] != e['frame_bits']:
                raise ValueError('Communication frame mismatch')
            elapsed = (rx['receive_complete_ns'] - e['send_start_ns']) / 1e9
            if elapsed < 0:
                raise ValueError('Negative delivery interval')
            # ARMOR timestamps start after serialization, unlike the Lancelot timer.
            elapsed += e.get('serialize_s', 0.)
            key = 'Client_Side_all_messages' if 'client' in (e['role'], e['peer']) else 'Server_Side_all_messages'
            row[key] += elapsed
            if e['role'] == 'client' and e['message_type'] == 'local_update':
                row['Client_Side_model_upload'] += elapsed
        row['Total_all_messages'] = sum(row[k] for k in
            ('Train', 'Enc', 'HE_comp', 'Dec', 'Client_Side_all_messages', 'Server_Side_all_messages'))
        rows.append(row)
    with (folder / 'rounds.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    mean = dict(dataset=dataset, measured_rounds=3)
    mean.update({k: sum(row[k] for row in rows[1:]) / 3 for k in rows[0]
                 if k not in {'dataset', 'round', 'measured'}})
    save(folder / 'mean_rounds2to4.json', mean)
    return mean


def stop(processes):
    for p in processes:
        if p.poll() is None:
            subprocess.run(['sudo', '-n', 'kill', '-TERM', '--', f'-{p.pid}'], check=False)
    deadline = time.monotonic() + 10
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(.2)
    for p in processes:
        if p.poll() is None:
            subprocess.run(['sudo', '-n', 'kill', '-KILL', '--', f'-{p.pid}'], check=False)
        p.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--replace-existing', action='store_true',
                        help='Delete this ARMOR output and its old comparison CSV before rerunning')
    parser.add_argument('--output', type=Path, default=ROOT / 'ARMOR_Lancelot_logs/armor_costs' /
                        ('rtt02_' + time.strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    output = args.output.resolve()
    plan = dict(status='planned', datasets=['MNIST', 'FMNIST', 'CIFAR10'], num_clients=30,
                training_timing_scope='lancelot_local_training_v1',
                total_rounds=4, excluded_rounds=[1], measured_rounds=[2, 3, 4], gpu=args.gpu,
                rate_mbps=3000, added_rtt_ms=.2, attack='No_Attack', attack_ratio=0,
                commands={ds: commands(args, output / ds.lower(), ds)
                          for ds in ('MNIST', 'FMNIST', 'CIFAR10')})
    if args.dry_run:
        plan['replace_existing'] = args.replace_existing
        print(json.dumps(plan, indent=2)); return
    # Authenticate in the caller's terminal session before starting isolated
    # role sessions. sudo's timestamp is normally scoped to that terminal;
    # sudo -v in the caller does not authenticate start_new_session children.
    if os.geteuid() != 0:
        os.execvp('sudo', ['sudo', '-n', '--', sys.executable, str(Path(__file__).resolve()),
                          '--gpu', str(args.gpu), '--output', str(output)] +
                  (['--replace-existing'] if args.replace_existing else []))
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    with (ROOT / 'ARMOR_Lancelot_logs/armor_costs/.rtt02.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Authenticate, acquire the experiment lock, and verify CUDA before
        # deleting old measurements. Never accept a replacement outside costs.
        import torch
        if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
            raise RuntimeError(f'GPU {args.gpu} unavailable; existing results retained')
        if args.replace_existing:
            costs = (ROOT / 'ARMOR_Lancelot_logs/armor_costs').resolve()
            if output.parent != costs or not output.name.startswith('rtt02_'):
                raise ValueError('--replace-existing requires an armor_costs/rtt02_* output')
            if output.exists():
                shutil.rmtree(output)
            (ROOT / 'ARMOR_Lancelot_logs/comparison_rtt02.csv').unlink(missing_ok=True)
        output.mkdir(parents=True, exist_ok=False)
        save(output / 'experiment.json', plan)
        created_ns, created_links, means = [], [], []
        try:
            import torch
            if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
                raise RuntimeError(f'GPU {args.gpu} unavailable; no CPU fallback')
            subprocess.run(['sudo', '-n', 'true'], check=True)
            names = [BRIDGE] + [f'ar02-{i}h' for i in range(4)]
            namespaces = subprocess.check_output(['sudo', '-n', 'ip', 'netns', 'list'], text=True)
            if any(f'ar02-{role}' in namespaces.split() for role in ROLES):
                raise RuntimeError('Owned namespace already exists; inspect before retrying')
            for name in names:
                if subprocess.run(['ip', 'link', 'show', name], capture_output=True).returncode == 0:
                    raise RuntimeError(f'Interface {name} already exists')
            os.environ['ARMOR_AUTHKEY'] = secrets.token_hex(32)
            for command in network_commands():
                subprocess.run(['sudo', '-n', *command], check=True, stdout=subprocess.DEVNULL)
                if command[:3] == ['ip', 'netns', 'add']: created_ns.append(command[3])
                if command[:3] == ['ip', 'link', 'add']: created_links.append(command[3])
            with (output / 'network_status.log').open('w') as f:
                for i, role in enumerate(ROLES):
                    subprocess.run(['sudo', '-n', 'tc', '-s', 'qdisc', 'show', 'dev', f'ar02-{i}h'],
                                   check=True, stdout=f, stderr=f)
                for sender, receiver in [('client', 'server1'), ('server1', 'server2')]:
                    result = subprocess.run(['sudo', '-n', 'ip', 'netns', 'exec', f'ar02-{sender}',
                                             'ping', '-c', '5', '-i', '.2', '-W', '2', ADDRESSES[receiver]],
                                            check=True, capture_output=True, text=True)
                    f.write(f'\n{sender} -> {receiver}\n' + result.stdout)
                    import re
                    match = re.search(r'= [\d.]+/([\d.]+)/', result.stdout)
                    if not match or float(match[1]) > 2:
                        raise RuntimeError('Measured ping RTT exceeds 2ms or missing; inspect network log')
            plan['status'] = 'running'; save(output / 'experiment.json', plan)
            for dataset in plan['datasets']:
                folder = output / dataset.lower(); folder.mkdir()
                plan['current_dataset'] = dataset; save(output / 'experiment.json', plan)
                processes, handles = [], []
                try:
                    roles = commands(args, folder, dataset)
                    def role_failures():
                        return [f'{roles[i][0]} exit={p.returncode} log={folder / (roles[i][0] + ".stdout.log")}'
                                for i, p in enumerate(processes) if p.poll() not in (None, 0)]
                    for role, command in roles:
                        handle = (folder / f'{role}.stdout.log').open('w'); handles.append(handle)
                        processes.append(subprocess.Popen(command, cwd=ROOT.parent, stdout=handle,
                                                          stderr=subprocess.STDOUT, start_new_session=True))
                        time.sleep(1)
                        if role_failures():
                            raise RuntimeError(f'{dataset}: ' + '; '.join(role_failures()))
                    deadline = time.monotonic() + 12 * 3600
                    while any(p.poll() is None for p in processes):
                        failed = role_failures()
                        if failed or time.monotonic() > deadline:
                            raise RuntimeError(f'{dataset}: role failed {failed} or timeout; inspect stdout logs')
                        time.sleep(2)
                    if any(p.returncode for p in processes):
                        raise RuntimeError(f'{dataset}: ' + '; '.join(role_failures()))
                    means.append(summarize(folder, dataset))
                    with (output / 'means_rounds2to4.csv').open('w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=means[0]); writer.writeheader(); writer.writerows(means)
                    print(f'{dataset}: completed; measured rounds 2-4', flush=True)
                finally:
                    stop(processes)
                    for handle in handles: handle.close()
            plan.update(status='completed', current_dataset=None)
        except BaseException as error:
            plan.update(status='failed', error=str(error) or type(error).__name__)
            raise
        finally:
            for ns in reversed(created_ns):
                subprocess.run(['sudo', '-n', 'ip', 'netns', 'del', ns], check=False)
            for name in reversed(created_links):
                subprocess.run(['sudo', '-n', 'ip', 'link', 'del', name], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            save(output / 'experiment.json', plan)


if __name__ == '__main__':
    main()
