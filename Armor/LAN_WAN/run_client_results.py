#!/usr/bin/env python3
"""Run sequential FMNIST LAN experiments and retain per-role, per-round results."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'ARMOR_client_results'
ROLES = ('keycenter', 'server2', 'server1', 'client')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def summarize(folder, clients, expected_rounds, dataset="FMNIST"):
    logs = {}
    for role in ROLES:
        path = folder / f'{dataset.lower()}_{role}.jsonl'
        logs[role] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    complete = [{e['round'] for e in logs[role] if e['event'] == 'round_summary'} for role in ('client', 'server1')]
    rows = []
    for r in sorted(complete[0] & complete[1]):
        es = {role: [e for e in events if e.get('round') == r] for role, events in logs.items()}
        def seconds(role, names):
            return sum(e['latency_s'] for e in es[role] if e['event'] in names)
        enc = [e for e in es['client'] if e['event'] == 'encryption']
        if len(enc) != clients or {e['client_id'] for e in enc} != set(range(clients)):
            raise ValueError(f'Round {r + 1}: incomplete client encryption records')
        evaluation = [e for e in es['server1'] if e['event'] == 'evaluation']
        if len(evaluation) != 1:
            raise ValueError(f'Round {r + 1}: missing or duplicate evaluation')
        ev = evaluation[0]
        row = dict(dataset=dataset, clients=clients, round=r + 1,
                   train_all_s=seconds('client', {'training'}),
                   enc_all_s=seconds('client', {'encryption'}),
                   he_comp_s=seconds('server1', {'norm_he', 'similarity_he', 'aggregation_he'}),
                   dec_total_compute_s=sum(seconds(role, {'decrypt_compute'}) for role in ('server1', 'server2')),
                   round_wall_s=seconds('server1', {'round_summary'}),
                   accuracy_pct=ev['accuracy'], loss=ev['loss'], survivors=len(ev['survivor_ids']))
        for phase in ('norm', 'similarity', 'aggregation'):
            row[f'{phase}_he_s'] = seconds('server1', {phase + '_he'})
            row[f'{phase}_dec_s'] = sum(e['latency_s'] for role in ('server1', 'server2')
                for e in es[role] if e['event'] == 'decrypt_compute' and e.get('phase') == phase)
        receivers = {e.get('message_id'): e for events in es.values() for e in events
                     if e['event'] == 'communication' and e.get('direction') == 'receive' and e.get('message_id')}
        pairs = {'client_upload': {('client', 'server1')},
                 'client_download': {('server1', 'client')},
                 'server_side': {('server1', 'server2'), ('server2', 'server1')}}
        for label, endpoints in pairs.items():
            sends = [e for events in es.values() for e in events if e['event'] == 'communication'
                     and e.get('direction') == 'send' and (e['role'], e.get('peer')) in endpoints
                     and e.get('message_type') in {'local_update', 'global_update', 'partial', 'partial_result'}]
            row[label + '_tx_s'] = sum(e['latency_s'] for e in sends)
            row[label + '_MB'] = sum(e['frame_bits'] for e in sends) / 8e6
            intervals = [(receivers.get(e.get('message_id')), e) for e in sends]
            row[label + '_delivery_s'] = (sum((receive['receive_complete_ns'] - send['send_start_ns']) / 1e9
                for receive, send in intervals) if all(receive and 'receive_complete_ns' in receive
                and 'send_start_ns' in send for receive, send in intervals) else None)
        rows.append(row)
    if rows:
        with (folder / 'runtime.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        numeric = [k for k in rows[0] if k not in {'dataset', 'clients', 'round'}]
        means = []
        for scope, chosen in [('mean_all', rows), ('mean_rounds2plus', rows[1:])]:
            if chosen:
                means.append(dict(dataset=dataset, clients=clients, scope=scope, rounds=len(chosen),
                    **{k: sum(x[k] for x in chosen) / len(chosen) if all(x[k] is not None for x in chosen) else None for k in numeric}))
        write_json(folder / 'runtime.json', dict(rows=rows, means=means,
            notes=['Times are seconds; MB is decimal.', 'Round numbers in this table are one-based; raw logs are zero-based.',
                   'TX includes serialization and blocking send. Delivery sums message intervals and is not wall time.',
                   'Blank delivery values indicate missing matched timing records, not zero.',
                   'Training and encryption sum all logical clients; a single process executes them sequentially.']))
    done = complete[0] == complete[1] == set(range(expected_rounds))
    write_json(folder / 'completion.json', dict(complete=done, completed_rounds=len(rows), expected_rounds=expected_rounds))
    return done


def stop(processes):
    live = [p for p in processes if p.poll() is None]
    for p in live:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except PermissionError:
            subprocess.run(['sudo', '-n', 'kill', '-TERM', '--', f'-{p.pid}'], check=False)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while any(p.poll() is None for p in live) and time.monotonic() < deadline:
        time.sleep(0.2)
    for p in live:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except PermissionError:
                subprocess.run(['sudo', '-n', 'kill', '-KILL', '--', f'-{p.pid}'], check=False)
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clients', nargs='+', type=int, choices=[30, 50, 70, 100], default=[30, 50, 70, 100])
    parser.add_argument('--python', default=os.environ.get('ARMOR_PYTHON', sys.executable))
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--timeout-hours', type=float, default=6)
    args = parser.parse_args()
    if args.rounds < 1 or args.timeout_hours <= 0:
        parser.error('rounds and timeout-hours must be positive')
    RESULTS.mkdir(exist_ok=True)
    for n in (30, 50, 70, 100):
        (RESULTS / f'Client{n}').mkdir(exist_ok=True)
    run_id = datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    env = dict(os.environ, ARMOR_PYTHON=args.python, DATASET='FMNIST', GLOBAL_EP=str(args.rounds),
               LOCAL_EP='1', LOCAL_BS='32', ATTACK='MPAF', ATTACK_RATIO='0.3', ATTACK_TIME='1', SEED='2023',
               PYTHONUNBUFFERED='1', CONNECT_TIMEOUT='600')
    # Role launcher uses an inherited auth key; never write it to experiment metadata.
    import secrets
    env['ARMOR_AUTHKEY'] = secrets.token_hex(32)
    def commands(n, folder):
        return [[ 'bash', str(ROOT/'lan.sh'), role] +
                (['--log_path', str(folder/'fmnist_{role}.jsonl')] if role == 'keycenter' else []) for role in ROLES]
    if args.dry_run:
        import shlex
        for n in args.clients:
            folder = RESULTS / f'Client{n}' / run_id
            print(f'FMNIST / LAN / {n} clients / {args.rounds} rounds -> {folder}')
            for command in commands(n, folder):
                print(shlex.join(command))
        return 0
    checks = [ ['nvidia-smi', '-L'], ['sudo', '-n', 'true'], ['ip', 'netns', 'list'],
               [args.python, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'] ]
    failures = []
    with (RESULTS/'preflight.log').open('w') as log:
        for command in checks:
            try:
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
                log.write(' '.join(command) + '\n' + result.stdout + '\n')
                if result.returncode:
                    failures.append(' '.join(command))
            except (OSError, subprocess.TimeoutExpired) as exc:
                log.write(str(exc) + '\n'); failures.append(' '.join(command))
    if failures:
        write_json(RESULTS/'launch_status.json', dict(status='preflight_failed', experiments_started=False,
            failed_checks=failures, log=str(RESULTS/'preflight.log')))
        print('Experiments not started: GPU/network/sudo preflight failed. See ' + str(RESULTS/'preflight.log'), file=sys.stderr)
        return 2
    # Set up only missing experiment namespaces; existing links are explicitly set to LAN.
    namespaces = subprocess.check_output(['sudo', '-n', 'ip', 'netns', 'list'], text=True)
    for link, ns in [('server2', 'armor-s2'), ('client', 'armor-client')]:
        action = 'profile' if any(line.split()[0] == ns for line in namespaces.splitlines() if line.split()) else 'setup'
        subprocess.run(['sudo', '-n', 'bash', str(ROOT/'network.sh'), '--link', link, action, 'lan'], check=True)
    write_json(RESULTS/'launch_status.json', dict(status='running', run_id=run_id, client_counts=args.clients))
    for n in args.clients:
        folder = RESULTS / f'Client{n}' / run_id
        folder.mkdir(parents=True, exist_ok=False)
        env.update(NUM_CLIENTS=str(n), RUN_TAG=run_id + f'_c{n}')
        config = dict(dataset='FMNIST', method='ARMOR', network='LAN', rate_mbps=3000, one_way_delay_ms=0.1,
            num_clients=n, global_rounds=args.rounds, local_epochs=1, batch_size=32,
            attack='MPAF', attack_ratio=0.3, attack_start_round=2, seed=2023,
            status='running', started_at=datetime.datetime.now().isoformat(), python=args.python)
        write_json(folder/'experiment.json', config)
        with (folder/'network_status.log').open('w') as log:
            subprocess.run(['sudo', '-n', 'bash', str(ROOT/'network.sh'), 'status'], stdout=log, stderr=subprocess.STDOUT, check=True)
        processes = [];streams = []
        try:
            for role, command in zip(ROLES, commands(n, folder)):
                stream = (folder/f'{role}.stdout.log').open('w');streams.append(stream)
                processes.append(subprocess.Popen(command, env=env, cwd=ROOT, stdout=stream,
                    stderr=subprocess.STDOUT, process_group=0))
            deadline = time.monotonic() + args.timeout_hours * 3600
            renewed = time.monotonic()
            while any(p.poll() is None for p in processes):
                bad = [(role, p.returncode) for role,p in zip(ROLES,processes) if p.poll() not in (None,0)]
                if bad:
                    raise RuntimeError(f'Role failure: {bad}')
                if time.monotonic() > deadline:
                    raise TimeoutError('Experiment exceeded timeout')
                if time.monotonic() - renewed > 60:
                    subprocess.run(['sudo', '-n', '-v'], check=True);renewed=time.monotonic()
                time.sleep(1)
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError('One or more roles failed')
            if not summarize(folder, n, args.rounds):
                raise RuntimeError('Round completion validation failed')
            config['status'] = 'completed'
        except BaseException as exc:
            stop(processes)
            config.update(status='failed', error=str(exc))
            try:
                summarize(folder,n,args.rounds)
            except Exception as summarize_error:
                config['summary_error']=str(summarize_error)
            write_json(RESULTS/'launch_status.json', dict(status='failed', client_count=n, folder=str(folder), error=str(exc)))
            raise
        finally:
            for stream in streams:stream.close()
            config['finished_at']=datetime.datetime.now().isoformat()
            write_json(folder/'experiment.json',config)
        print(f'Completed {n} clients: {folder}',flush=True)
    write_json(RESULTS/'launch_status.json', dict(status='completed',run_id=run_id,client_counts=args.clients))
    return 0


if __name__ == '__main__':
    sys.exit(main())
