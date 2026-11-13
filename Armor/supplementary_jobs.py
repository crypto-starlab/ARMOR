"""Run missing supplementary experiments, saving only standard accuracy text logs."""
import argparse
import datetime
from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from log_paths import accuracy_log_path, normalize_attack as normalize_log_attack

ARMOR_DIR=Path(__file__).resolve().parent
OUTPUT_ROOT=ARMOR_DIR/'accuracy_logs/SupplyMentary'
STOP_FILE=OUTPUT_ROOT/'.stop_after_current'
DATASETS=('MNIST','FMNIST','CIFAR10')
ATTACKS=('alie','sign_flipping','label_flipping','minsum','minmax')
METHODS=('armor','vanilla_armor','fedavg','trimmed_mean','cosine','multi_krum','krum','median')
ALIASES={'min_sum':'minsum','min_max':'minmax','sign_flip':'sign_flipping','signflipping':'sign_flipping',
         'label_flip':'label_flipping','labelflipping':'label_flipping','alie_attack':'alie'}
METHOD_ALIASES={'plain_armor':'vanilla_armor','vanila_armor':'vanilla_armor','vanila':'vanilla_armor'}
ROUND=re.compile(r'^\s*(\d+)\s+Round acc\s*:\s*([^|\s]+)\s*\|\s*Loss:\s*(\S+)',re.MULTILINE)


def normalize_attack(name):
    name=str(name).lower().replace('-','_').replace(' ','_')
    name=ALIASES.get(name,name)
    if name not in ATTACKS:raise ValueError('Supplementary attacks: '+', '.join(ATTACKS))
    return name


def accuracy_log(dataset,method,attack,root=None):
    if dataset not in DATASETS or method not in METHODS:raise ValueError('Unsupported dataset or method')
    return accuracy_log_path(dataset,method,attack,root=root)


def has_complete_run(path,rounds=200,require_finite=False):
    if not Path(path).is_file():return False
    expected=1
    for match in ROUND.finditer(Path(path).read_text(errors='replace')):
        number=int(match[1])
        try:finite=math.isfinite(float(match[2])) and math.isfinite(float(match[3]))
        except ValueError:finite=False
        if require_finite and not finite:expected=1;continue
        if number==expected:expected+=1
        elif number==1:expected=2
        else:expected=1
        if expected==rounds+1:return True
    return False


def command_for(dataset,method,attack,gpu=0):
    return [sys.executable,str(ARMOR_DIR/'main.py'),'--dataset',dataset,
        '--method','armor' if method=='vanilla_armor' else method,
        *(['--cipher'] if method=='armor' else []),
        '--attack',attack,'--attack_ratio','0.3','--attack_time','50',
        '--num_clients','50','--global_ep','200','--local_ep','1','--local_bs','32','--seed','2023',
        '--gpu',str(gpu),'--log_path',str(accuracy_log_path(dataset,method,attack))]


def inventory(datasets=DATASETS,methods=METHODS,attacks=ATTACKS,root=None):
    root=OUTPUT_ROOT if root is None else root
    rows=[]
    for dataset in datasets:
        for attack in attacks:
            for method in methods:
                path=accuracy_log(dataset,method,attack,root)
                rows.append(dict(dataset=dataset,attack=attack,method=method,
                    status='complete' if has_complete_run(path) else 'incomplete' if path.exists() else 'missing',log_path=str(path)))
    return rows


@contextmanager
def experiment_output(args):
    """Write only the standard accuracy text file; restart incomplete runs in place."""
    # Existing workers import this module anew in each experiment subprocess.
    # Code 75 is already understood by older workers as a clean skip.
    if STOP_FILE.exists():
        print(f'Skipping new experiment: stop-after-current requested ({STOP_FILE})',flush=True)
        raise SystemExit(75)
    args.dataset=args.dataset.upper()
    args.attack=normalize_log_attack(args.attack)
    method='armor' if args.cipher else 'vanilla_armor' if args.method=='armor' else args.method
    target=Path(args.log_path) if args.log_path else ARMOR_DIR/accuracy_log_path(args.dataset,method,args.attack,args.attack_ratio).name
    target.parent.mkdir(parents=True,exist_ok=True)
    # Keep a stable lock file per experiment so GPUs can share the queue.
    lock=os.open(str(target)+'.lock',os.O_CREAT | os.O_RDWR,0o600)
    try:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'Skipping experiment running on another worker: {target}',flush=True)
            raise SystemExit(75)
        if target.exists():
            archive=target.parent/'archive'
            archive.mkdir(exist_ok=True)
            stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            backup=archive/f'{target.stem}_{stamp}_{os.getpid()}{target.suffix}'
            target.rename(backup)
            print(f'Archived previous log: {backup}',flush=True)
        require_finite = not (args.method == 'fedavg' and args.attack == 'mpaf')
        target.write_text('')
        args.log_path=str(target)
        yield target
        if not has_complete_run(target,args.global_ep,require_finite=require_finite):
            raise RuntimeError(f'Incomplete accuracy log: {target}')
        print(f'Completed: {target}',flush=True)
    finally:
        os.close(lock)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true',help='List missing jobs without running training')
    parser.add_argument('--rerun',action='store_true',help='Archive selected existing logs under their experiment lock and rerun them')
    parser.add_argument('--resume',action='store_true',help='Clear a previous stop-after-current request')
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    parser.add_argument('--attacks',nargs='+',default=list(ATTACKS))
    parser.add_argument('--methods',nargs='+',default=list(METHODS))
    parser.add_argument('--gpu',type=int,default=int(os.environ.get('GPU','0')))
    args=parser.parse_args()
    try:
        attacks=[normalize_attack(x) for x in args.attacks]
        methods=[METHOD_ALIASES.get(x.lower(),x.lower()) for x in args.methods]
        if any(x not in METHODS for x in methods):raise ValueError('Methods: '+', '.join(METHODS))
    except ValueError as exc:parser.error(str(exc))
    rows=inventory(args.datasets,methods,attacks)
    pending=[r for r in rows if args.rerun or r['status'] in {'missing','incomplete'}]
    print(f'Experiments: {len(rows)}; existing full-length logs: {sum(r['status']=='complete' for r in rows)}; remaining: {len(pending)}',flush=True)
    for dataset in args.datasets:
        chosen=[r for r in rows if r['dataset']==dataset]
        print(f'{dataset}: {sum(args.rerun or r["status"] in {"missing","incomplete"} for r in chosen)} remaining / {len(chosen)}',flush=True)
    if args.dry_run:
        for row in pending:
            print(shlex.join(command_for(row['dataset'],row['method'],row['attack'],args.gpu)))
        return 0
    if args.resume:STOP_FILE.unlink(missing_ok=True)
    if STOP_FILE.exists():
        print('Stop-after-current request is active; use --resume to start more experiments.',flush=True)
        return 0
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    environment['ARMOR_SUPPLEMENTARY_RERUN']='1' if args.rerun else '0'
    for i,row in enumerate(pending,1):
        if STOP_FILE.exists():
            print('Stop-after-current requested; worker finished.',flush=True)
            return 0
        target=Path(row['log_path'])
        if not args.rerun and has_complete_run(target):continue
        target.parent.mkdir(parents=True,exist_ok=True)
        command=command_for(row['dataset'],row['method'],row['attack'],args.gpu)
        print(f'[{i}/{len(pending)}] {row["dataset"]} / {row["attack"]} / {row["method"]}',flush=True)
        process=subprocess.Popen(command,cwd=ARMOR_DIR,env=environment)
        try:
            code=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
        if code==75:continue
        if code or not has_complete_run(target):
            print(f'Experiment failed. Accuracy log: {target}',file=sys.stderr)
            return code or 1
    print('Worker finished; experiments held by other workers may still be running.',flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
