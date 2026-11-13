"""Summarize measured norm runs without double-counting mirrored network logs."""
import argparse
import csv
import json
from pathlib import Path


def read_summary(path):
    data=json.loads(path.read_text())
    record=dict(method=data['method'],run_tag=data['config']['run_tag'],
                input_blocks=data.get('input_blocks'),elements=data.get('elements'),
                chunks_per_client=data.get('chunks_per_client'),input_sha256=data.get('input_sha256'),
                samples=data['samples'],
                all_valid=data['all_valid'],max_absolute_error=data['max_absolute_error'])
    for name in ('client_upload_bytes','s1_s2_tx_bytes','s1_s2_rx_bytes','jobs'):
        record[name]=data[name]['mean']
    for name in ('client_encrypt_s','he_s','rpc_s','local_decrypt_decode_s','server2_compute_s','norm_total_s'):
        record[name.replace('_s','_ms')]=data[name]['mean']*1000
    record['s1_s2_total_bytes']=record['s1_s2_tx_bytes']+record['s1_s2_rx_bytes']
    record['keycenter_tx_bytes']=record['keycenter_rx_bytes']=0
    record['key_generation_ms']=None
    key_log=(path.parent.parent if path.parent.name.startswith('blocks_') else path.parent)/'keycenter.jsonl'
    if key_log.exists():
        for line in key_log.read_text().splitlines():
            event=json.loads(line)
            if event['event']=='communication':
                field='keycenter_tx_bytes' if event['direction']=='send' else 'keycenter_rx_bytes'
                record[field]+=event['frame_bits']/8
            elif event['event']=='key_generation':
                record['key_generation_ms']=event['latency_s']*1000
    record['summary_path']=str(path)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path,default=Path(__file__).resolve().parent/'runs/norm_benchmarks')
    parser.add_argument('--csv',type=Path)
    args=parser.parse_args()
    paths=list(args.directory.glob('*/*/blocks_*/summary.json'))
    rows=[read_summary(p) for p in sorted(paths)]
    if not rows:
        parser.error('No METHOD/RUN_TAG/blocks_*/summary.json files found')
    print('method          run_tag              blocks  elements samples valid  upload_MB  s1_s2_MB     HE_ms    RPC_ms  local_dec_ms  norm_total_ms')
    for r in rows:
        print(f'{r["method"]:15s} {r["run_tag"]:20s} {str(r["input_blocks"]):>6s} {str(r["elements"]):>9s} {r["samples"]:7d} {str(r["all_valid"]):5s} '
              f'{r["client_upload_bytes"]/1e6:10.4f} {r["s1_s2_total_bytes"]/1e6:9.4f} '
              f'{r["he_ms"]:9.3f} {r["rpc_ms"]:9.3f} {r["local_decrypt_decode_ms"]:13.3f} {r["norm_total_ms"]:14.3f}')
    if args.csv:
        args.csv.parent.mkdir(parents=True,exist_ok=True)
        with args.csv.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    main()
