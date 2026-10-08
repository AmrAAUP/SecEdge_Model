from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,statistics
R=Path(__file__).resolve().parent
sha=lambda x:hashlib.sha256(x).hexdigest()
p=json.loads((R/'PROTOCOL.json').read_text(encoding='utf-8'))
assert sha((R/'PROTOCOL.json').read_bytes())==(R/'PROTOCOL.sha256').read_text().split()[0]
events=[json.loads(x) for x in (R/'service_run.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
assert events[0]['event']=='run_started' and events[-1]['event']=='run_closed' and events[-1]['complete']
assert events[-1]['service_thread_alive'] is False
assert events[0]['protocol_sha256']==sha((R/'PROTOCOL.json').read_bytes())
levels=['None','Low','Moderate','High'];order={s:i for i,s in enumerate(levels)}
def tier(k,n):return 'None' if k==0 else 'Low' if 10*k<=n else 'Moderate' if 10*k<=3*n else 'High'
def checks(x):
 expected=hashlib.sha256((x['nonce']+'|service-v1').encode()).hexdigest()
 correct=x['returned_nonce']==x['nonce'] and x['returned_digest']==expected
 transport=x['status']==200 and x['error'] is None and x['duration_ns']<=p['request_slo_ms']*1000000
 assert x['body_correct']==correct
 assert x['transport_slo_success']==transport
 assert x['functional_slo_success']==(transport and correct)
 return transport,transport and correct
records=[e for e in events if e['event']=='request']
assert len(records)==1344 and len({x['nonce'] for x in records})==1344
for x in records:checks(x)
episodes=[]
for spec in p['episode_order']:
 eid=spec['episode'];rr=[x for x in records if x['episode']==eid]
 phase={k:sorted([x for x in rr if x['phase']==k],key=lambda x:x['request_index']) for k in ['monitor','future_client','recovery']}
 for k,n in [('monitor',16),('future_client',24),('recovery',8)]:
  assert len(phase[k])==n and [x['request_index'] for x in phase[k]]==list(range(n))
 fc=[e for e in events if e['event']=='forecast_committed' and e['episode']==eid];assert len(fc)==1;fc=fc[0]
 assert max(x['finished_elapsed_ns'] for x in phase['monitor'])<fc['elapsed_ns']<min(x['started_elapsed_ns'] for x in phase['future_client'])
 pred={key:tier(sum(not x[success] for x in phase['monitor']),16) for key,success in [('functional','functional_slo_success'),('transport_only','transport_slo_success')]}
 for key in pred:assert pred[key]==fc['forecast'][key]
 bad=sum(not x['functional_slo_success'] for x in phase['future_client']);truth=tier(bad,24)
 completed=[e for e in events if e['event']=='episode_completed' and e['episode']==eid];assert len(completed)==1
 assert completed[0]['observed_tier']==truth and completed[0]['future_functional_bad']==bad
 episodes.append({'episode':eid,'block':spec['block'],'condition':spec['id'],'scope':'transition' if spec['id'] in ('abrupt_onset','recovery_transition') else 'stationary','predicted':pred,'observed':truth,'future_bad':bad,'future_n':24,'recovery_bad':sum(not x['functional_slo_success'] for x in phase['recovery']),'recovery_n':8,'monitor_functional_bad':sum(not x['functional_slo_success'] for x in phase['monitor']),'monitor_transport_bad':sum(not x['transport_slo_success'] for x in phase['monitor'])})
def metrics(rows,key):
 cm=[[0]*4 for _ in range(4)]
 for row in rows:cm[order[row['observed']]][order[row['predicted'][key]]]+=1
 exact=sum(x['observed']==x['predicted'][key] for x in rows)
 high=[x for x in rows if x['observed']=='High']
 return {'n':len(rows),'exact':exact,'agreement':exact/len(rows),'under_triage':sum(order[x['predicted'][key]]<order[x['observed']] for x in rows),'over_triage':sum(order[x['predicted'][key]]>order[x['observed']] for x in rows),'high_n':len(high),'high_correct':sum(x['predicted'][key]=='High' for x in high),'mean_absolute_tier_error':sum(abs(order[x['predicted'][key]]-order[x['observed']]) for x in rows)/len(rows),'confusion_rows_observed_columns_predicted':cm}
summary={scope:{key:metrics(rows,key) for key in ['functional','transport_only']} for scope,rows in [('all',episodes),('stationary',[r for r in episodes if r['scope']=='stationary']),('transition',[r for r in episodes if r['scope']=='transition'])]}
bycondition={c['id']:{'n':len(rows),'future_failed':sum(x['future_bad'] for x in rows),'future_n':sum(x['future_n'] for x in rows),'observed_tier_counts':dict(Counter(x['observed'] for x in rows)),**{key:metrics(rows,key) for key in ['functional','transport_only']}} for c in p['conditions'] for rows in [[r for r in episodes if r['condition']==c['id']]]}
out={'title':p['title'],'protocol_sha256':sha((R/'PROTOCOL.json').read_bytes()),'worker_sha256':sha((R/'service_worker.py').read_bytes()),'raw_sha256':sha((R/'service_run.jsonl').read_bytes()),'request_count':len(records),'episode_count':len(episodes),'levels':levels,'summary':summary,'per_condition':bycondition,'observed_tier_counts':dict(Counter(x['observed'] for x in episodes)),'recovery_success':sum(x['functional_slo_success'] for x in records if x['phase']=='recovery'),'recovery_n':224,'forecast_precedes_every_future_request':True,'episodes':episodes,'metadata':events[0],'closure':events[-1]}
(R/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'summary':summary,'per_condition':bycondition,'observed':out['observed_tier_counts'],'recovery':[out['recovery_success'],224]},indent=2))
