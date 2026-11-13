import json,heapq,collections,pathlib,time
workspace=pathlib.Path(__file__).resolve().parents[5]
root=workspace/'siyang-jiang-Lancelot-Dev-ea2f9eb/logs/oringinal_flow_rep1'
out=workspace/'analysis/lancelot_costs'
reports=[]
for dataset in ['mnist','fmnist','cifar10']:
    sums=collections.defaultdict(float); counts=collections.Counter(); comm={}; pending={}; config=None
    def records(path):
        with path.open() as f:
            for line in f:
                if line.strip(): yield json.loads(line)
    streams=[records(root/f'{dataset}_{role}.jsonl') for role in ['client','server','keycenter']]
    for e in heapq.merge(*streams,key=lambda e:e['timestamp']):
        ev=e['event']; role=e['role']; kind=e.get('kind','')
        if ev=='config' and role=='server': config=e
        if e.get('round') is None: continue
        if ev!='communication':
            k=f'{role}/{ev}/{kind}'; sums[k]+=e.get('latency_s',0); counts[k]+=1
            continue
        sender,receiver=(role,e['peer']) if e['direction']=='send' else (e['peer'],role)
        label=f"{sender}->{receiver}/{e['message_type']}"
        row=comm.setdefault(label,dict(MB=0.,tx_s=0.,rx_io_s=0.,download_s=0.,messages=0,matched=0))
        if e['direction']=='send':
            row['MB']+=e['frame_bits']/8e6; row['tx_s']+=e['latency_s']; row['messages']+=1
        else: row['rx_io_s']+=e.get('io_s',0.)
        key=(sender,receiver,e.get('message_sequence'))
        other=pending.pop(key,None)
        if other is None: pending[key]=e; continue
        tx,rx=(e,other) if e['direction']=='send' else (other,e)
        if tx.get('clock_domain')==rx.get('clock_domain') and tx.get('frame_bits')==rx.get('frame_bits') and 'transfer_start_s' in tx and 'download_complete_s' in rx:
            row['download_s']+=rx['download_complete_s']-tx['transfer_start_s']; row['matched']+=1
    report=dict(dataset=dataset,config=config,timings=dict(sums),counts=dict(counts),communication=comm,unmatched=len(pending))
    reports.append(report)
    (out/'costs.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(report),flush=True)
