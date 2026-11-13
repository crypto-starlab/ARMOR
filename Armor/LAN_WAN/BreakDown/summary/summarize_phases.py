import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
settings=[('MNIST','MPAF_LAN_20261111_125601_257751',50,60),('FMNIST','MPAF_LAN_20261111_125601_257751',50,60),('CIFAR10','MPAF_LAN_20261111_145856_800114',1,10)]
rows=[]; means=[]; sources=[]
for dataset,run,first,last in settings:
 folder=ROOT/dataset/run
 events=[]
 for role in ('client','server1','server2'):
  path=folder/f'{dataset.lower()}_{role}.jsonl'
  sources.append(str(path))
  with path.open() as stream:
   for line in stream:
    e=json.loads(line)
    if e.get('round') is not None and first-1<=e['round']<last:events.append(e)
 for role in ('client','server1'):
  complete=[e['round'] for e in events if e['role']==role and e['event']=='round_summary']
  assert sorted(complete)==list(range(first-1,last)),(dataset,role,complete)
 rx={e['message_id']:e for e in events if e['event']=='communication' and e['direction']=='receive'}
 for phase in ('norm','similarity','aggregation'):
  phase_rows=[]
  for r in range(first-1,last):
   es=[e for e in events if e.get('round')==r]
   he=[e for e in es if e['role']=='server1' and e['event']==phase+'_he']
   dec=[e for e in es if e['event']=='decrypt_compute' and e.get('phase')==phase]
   tx=[e for e in es if e['event']=='communication' and e['direction']=='send' and e.get('phase')==phase and {e['role'],e['peer']}=={'server1','server2'}]
   if he:
    assert len(he)==1 and {e['role'] for e in dec}=={'server1','server2'}
    assert len(tx)==2,(dataset,r,phase,len(tx))
   else:assert dataset=='CIFAR10' and r==0 and phase!='aggregation'
   comm=0
   for e in tx:
    receive=rx[e['message_id']]
    assert e['frame_bits']==receive['frame_bits']
    duration=(receive['receive_complete_ns']-e['send_start_ns'])/1e9+e.get('serialize_s',0)
    assert duration>=0
    comm+=duration
   row=dict(dataset=dataset,round=r+1,phase=phase,executed=bool(he),HEcomp_s=sum(e['latency_s'] for e in he),Dec_s=sum(e['latency_s'] for e in dec),communication_s=comm)
   rows.append(row);phase_rows.append(row)
  mean=dict(dataset=dataset,rounds=f'{first}-{last}',phase=phase,round_count=len(phase_rows),executed_rounds=sum(x['executed'] for x in phase_rows))
  mean.update({k:sum(x[k] for x in phase_rows)/len(phase_rows) for k in ('HEcomp_s','Dec_s','communication_s')})
  means.append(mean)
for name,data in [('phase_rounds.csv',rows),('phase_means.csv',means)]:
 with (ROOT/'summary'/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(ROOT/'summary'/'phase_sources.json').write_text(json.dumps(sources,indent=2))
for x in means:print(x)
