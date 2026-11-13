import json,pathlib,csv
root=pathlib.Path(__file__).resolve().parent
reports=json.loads((root/'costs.json').read_text())
rows=[]
for r in reports:
 t=r['timings']; c=r['communication']; get=lambda k:t.get(k,0.)
 row=dict(dataset=r['dataset'].upper(),rounds=r['counts']['server/round_summary/'],HE_distance_s=get('server/distance_evaluation/'),HE_aggregation_s=get('server/aggregation_evaluation/'),Enc_client_s=get('client/encryption/local_model'),Enc_auxiliary_s=get('server/encryption/auxiliary'),Enc_selector_s=get('keycenter/encryption/selector'),Dec_scores_s=get('keycenter/decryption/scores'),Dec_aggregate_s=get('keycenter/decryption/aggregate'),Encoding_client_s=get('client/encoding/local_model'),Selection_combined_s=get('keycenter/selection/'))
 row['HE_total_s']=row['HE_distance_s']+row['HE_aggregation_s']; row['Enc_total_s']=sum(row[k] for k in ['Enc_client_s','Enc_auxiliary_s','Enc_selector_s']); row['Dec_total_s']=row['Dec_scores_s']+row['Dec_aggregate_s']
 for field in ['MB','download_s','tx_s','messages','matched']: row['comm_'+field]=sum(x[field] for x in c.values())
 # Every request and response in the three operation RPC pairs.
 rpc={'encode_chunk','encoded_chunk','encrypt_chunk','encrypted_chunk','decrypt_chunk','decoded_chunk'}
 for field in ['MB','download_s']: row['comm_without_operation_RPC_'+field]=sum(x[field] for key,x in c.items() if key.split('/')[-1] not in rpc)
 rows.append(row)
with (root/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
print(json.dumps(rows,indent=2))
