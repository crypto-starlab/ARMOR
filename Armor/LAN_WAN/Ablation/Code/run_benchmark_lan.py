#!/usr/bin/env python3
"""Run one norm-computation method with consistent LAN settings and result files."""
import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parent
FIGURE=CODE.parent
NETWORK=FIGURE.parent/'network.sh'
METHODS={'wo_Dist':'w/o Dist','wo_Lazy':'w/o Lazy','wo_Coeff':'w/o Coeff','wo_Ext':'w/o Extraction','ARMOR':'ARMOR'}


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2)+'\n')


def write_results(folder,method,ringdim):
    events=[json.loads(line) for line in (folder/'client.jsonl').read_text().splitlines() if line.strip()]
    rows=[]
    for i,blocks in enumerate([62,184,367],1):
        summary=json.loads((folder/f'blocks_{blocks:04d}/summary.json').read_text())
        if summary['method']!=method or summary['samples']!=5 or not summary['all_valid']:
            raise ValueError(f'Invalid summary for {method}, {blocks} blocks')
        width=ringdim//2 if method=='wo_Coeff' else ringdim
        chunks=(blocks*16384+width-1)//width
        if summary['spec']['ringdim']!=ringdim or summary['chunks_per_client']!=chunks:
            raise ValueError('Unexpected ring dimension or chunk count')
        jobs=chunks if method=='wo_Lazy' else 1
        if summary['jobs']['mean']!=jobs:
            raise ValueError(f'Unexpected job schedule: {method}')
        uploads=[e['latency_s'] for e in events if e['event']=='upload' and e.get('input_blocks')==blocks and e.get('warmup') is False]
        if len(uploads)!=5:raise ValueError('Incomplete measured uploads')
        ms=lambda key:summary[key]['mean']*1000
        upload=sum(uploads)/5*1000
        rows.append(dict(Parameter=f'Param{i}',input_blocks=blocks,method=method,ringdim=ringdim,ciphertext_chunks=chunks,padding_elements=chunks*width-blocks*16384,
             Enc_ms=ms('client_encrypt_s'),HE_computation_ms=ms('he_s'),
             Dec_ms=ms('local_decrypt_decode_s')+ms('server2_compute_s'),
             Client_upload_ms=upload,Server_communication_ms=ms('rpc_s')-ms('server2_compute_s'),
             Total_ms=ms('client_encrypt_s')+upload+ms('norm_total_s')))
    with (folder/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    subprocess.run([sys.executable,str(CODE/'tools/render_results.py'),'--method',method,'--csv',str(folder/'results.csv'),'--output',str(folder/'results.png')],check=True)


def terminate(processes):
    for process in processes:
        if process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except PermissionError:subprocess.run(['sudo','-n','kill','-TERM','--',f'-{process.pid}'],check=False)
            except ProcessLookupError:pass
    deadline=time.monotonic()+10
    while any(p.poll() is None for p in processes) and time.monotonic()<deadline:time.sleep(.2)
    for process in processes:
        if process.poll() is None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except PermissionError:subprocess.run(['sudo','-n','kill','-KILL','--',f'-{process.pid}'],check=False)
            except ProcessLookupError:pass


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=list(METHODS),required=True)
    parser.add_argument('--run-tag',help='Fresh result directory name for a repeat run')
    parser.add_argument('--ringdim',type=int,choices=[16384,32768,65536],default=16384)
    parser.add_argument('--timeout',type=float,default=14400,help='Maximum seconds per method/ringdim')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    method=args.method
    tag=args.run_tag or f'results_{args.ringdim}'
    if Path(tag).name!=tag or tag in {'.','..'}:
        parser.error('--run-tag must be a single directory name')
    folder=FIGURE/method/tag
    python=os.environ.get('ARMOR_PYTHON',sys.executable)
    roles=['keycenter','server2','server1','client']
    port_base=19240+10*list(METHODS).index(method)
    ports=[port_base,port_base+1,port_base+2]
    env=dict(os.environ,ARMOR_PYTHON=python,ARMOR_AUTHKEY=secrets.token_hex(32),RUN_TAG=tag,
             KEYCENTER_HOST='10.203.77.1',KEYCENTER_BIND='10.203.77.1',
             SERVER1_HOST='127.0.0.1',SERVER1_BIND='127.0.0.1',SERVER2_HOST='10.203.77.2',SERVER2_BIND='10.203.77.2',
             KEYCENTER_PORT=str(ports[0]),SERVER1_PORT=str(ports[1]),SERVER2_PORT=str(ports[2]),
             PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    commands=[[ 'bash',str(CODE/'run_norm_role.sh'),method,role,'--profile','lan',
                '--output',str(FIGURE),'--run_tag',tag]+
              (['--ringdim',str(args.ringdim),'--input_blocks','62','184','367','--warmup','1','--iterations','5','--num_clients','1','--seed','2023']
               if role=='keycenter' else []) for role in roles]
    if args.dry_run:
        import shlex
        print(f'{METHODS[method]} -> {folder}')
        for command in commands:print(shlex.join(command))
        return 0
    with open('/tmp/armor_fig4_lan.lock','a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another Fig4 LAN experiment is running.',file=sys.stderr);return 1
        if folder.exists() and any(folder.iterdir()):
            status_file=folder/'experiment.json'
            previous=json.loads(status_file.read_text()) if status_file.exists() else {}
            if previous.get('status')=='completed' and previous.get('all_valid'):
                if (folder/'results.csv').is_file() and all((folder/f'blocks_{blocks:04d}/summary.json').is_file() for blocks in [62,184,367]):
                    print(f'Already completed {METHODS[method]}: {folder}; skipping.',flush=True)
                    return 0
            entries=[p for p in folder.iterdir() if p.name!='attempts']
            only_preflight=all(p.is_file() and p.name in {'experiment.json','preflight.log','network_status.log'} for p in entries)
            if previous.get('status')=='failed' and previous.get('experiments_started') is False and only_preflight:
                import shutil
                attempts=folder/'attempts';attempts.mkdir(exist_ok=True)
                number=1
                while (attempts/f'attempt_{number}').exists():number+=1
                archived=attempts/f'attempt_{number}';archived.mkdir()
                for path in entries:shutil.move(str(path),str(archived/path.name))
            else:
                print(f'Existing measurements are preserved: {folder}. This run was not started.',file=sys.stderr)
                return 1
        folder.mkdir(parents=True,exist_ok=True)
        status=dict(method=method,label=METHODS[method],status='preflight',experiments_started=False,ports=dict(keycenter=ports[0],server1=ports[1],server2=ports[2]),
                    profile='LAN',rate_mbps=3000,one_way_delay_ms=.1,shaped_link='Server1-Server2',
                    ringdim=args.ringdim,padding='Zero pad final ciphertext chunk; input length fixed',client_link='Host loopback; unshaped',input_blocks=[62,184,367],input_block_size=16384,
                    warmup=1,iterations=5,num_clients=1,seed=2023,python=python,
                    started_at=datetime.datetime.now().isoformat())
        save(folder/'experiment.json',status)
        processes=[];streams=[]
        try:
            with (folder/'preflight.log').open('w') as log:
                for command in [['nvidia-smi','-L'],['sudo','-n','true'],
                                [python,str(CODE/'check_native.py'),'--method',method,'--ringdim',str(args.ringdim)]]:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
            for port in ports:
                with socket.socket() as probe:
                    probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                    probe.bind(('127.0.0.1',port))
            namespaces=subprocess.check_output(['sudo','-n','ip','netns','list'],text=True)
            exists=any(line.split()[0]=='armor-s2' for line in namespaces.splitlines() if line.split())
            action='profile' if exists else 'setup'
            with (folder/'network_status.log').open('w') as log:
                subprocess.run(['sudo','-n','bash',str(NETWORK),'--link','server2',action,'lan'],check=True,stdout=log,stderr=subprocess.STDOUT)
                subprocess.run(['sudo','-n','bash',str(NETWORK),'--link','server2','status'],check=True,stdout=log,stderr=subprocess.STDOUT)
            status.update(status='running',experiments_started=True);save(folder/'experiment.json',status)
            for role,command in zip(roles,commands):
                stream=(folder/f'{role}.stdout.log').open('w');streams.append(stream)
                processes.append(subprocess.Popen(command,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,process_group=0))
            deadline=time.monotonic()+args.timeout;refreshed=time.monotonic()
            while any(p.poll() is None for p in processes):
                failed=[(r,p.returncode) for r,p in zip(roles,processes) if p.poll() not in (None,0)]
                if failed:raise RuntimeError(f'Role failure: {failed}')
                if time.monotonic()>deadline:raise TimeoutError(f'Benchmark exceeded {args.timeout} seconds')
                if time.monotonic()-refreshed>60:
                    subprocess.run(['sudo','-n','-v'],check=True);refreshed=time.monotonic()
                time.sleep(1)
            if any(p.returncode for p in processes):raise RuntimeError('A benchmark role failed')
            write_results(folder,method,args.ringdim)
            status.update(status='completed',all_valid=True)
            print(f'Completed {METHODS[method]}: {folder}',flush=True)
            return 0
        except BaseException as exc:
            status.update(status='failed',error=str(exc))
            print(f'Not completed: {exc}; logs: {folder}',file=sys.stderr)
            return 1
        finally:
            terminate(processes)
            for stream in streams:stream.close()
            status['finished_at']=datetime.datetime.now().isoformat();save(folder/'experiment.json',status)


if __name__=='__main__':sys.exit(main())
