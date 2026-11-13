import json,csv,pathlib
root=pathlib.Path(__file__).resolve().parent
rows=[]
for r in json.loads((root/'costs.json').read_text()):
 t=r['timings']; c=r['communication']; get=lambda k:t.get(k,0.)
 row=dict(dataset=r['dataset'].upper(),Train=get('client/training/'),Enc=sum(get(k) for k in ['client/encryption/local_model','server/encryption/auxiliary','keycenter/encryption/selector']),HE_comp=sum(get(k) for k in ['server/distance_evaluation/','server/aggregation_evaluation/']),Dec=sum(get(k) for k in ['keycenter/decryption/scores','keycenter/decryption/aggregate']))
 row['Client_Side_all_messages']=sum(v['download_s'] for k,v in c.items() if 'client' in k)
 row['Server_Side_all_messages']=sum(v['download_s'] for k,v in c.items() if 'client' not in k)
 row['Total_all_messages']=sum(row[k] for k in ['Train','Enc','HE_comp','Dec','Client_Side_all_messages','Server_Side_all_messages'])
 row['Client_Side_model_upload']=c['client->server/encrypted_chunk']['download_s']
 row['round_wall_client_s']=get('client/round_summary/')
 row['round_wall_server_s']=get('server/round_summary/')
 row['unmatched']=r['unmatched']
 rows.append(row)
with (root/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
