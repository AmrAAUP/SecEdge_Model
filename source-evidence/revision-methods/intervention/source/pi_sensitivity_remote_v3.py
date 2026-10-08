
import os,json,time,hashlib,math
from pathlib import Path
import numpy as np
from llama_cpp import Llama

def emit(event,**kw):
 print(json.dumps(dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),execution_phase='v3',execution_amendment_sha256=P['execution_amendment_sha256'],**kw)),flush=True)
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def eventkey(event,flow_id,donor_index=None,detail=None):
 values=[event,flow_id]
 if event in ('single','joint'):values.extend([donor_index,detail])
 return json.dumps(values,separators=(',',':'))
def oldkey(row):
 kind=row['event']
 if kind in ('baseline','flow_complete'):return eventkey(kind,row['flow_id'])
 if kind=='single':return eventkey(kind,row['flow_id'],row['donor_index'],row['field'])
 if kind=='joint':return eventkey(kind,row['flow_id'],row['donor_index'],row['method'])
 return None
prior={oldkey(r):r for r in H if oldkey(r) is not None}
assert len(prior)==len([r for r in H if oldkey(r) is not None])
logged_keys=set(prior)
completed_flows={r['flow_id'] for r in H if r['event']=='flow_complete'}
prior_baselines={r['flow_id']:r for r in H if r['event']=='baseline'}
historical_calls=sum(bool(r.get('executed')) for r in H if r['event'] in ('baseline','single','joint'))
historical_hashes={r['input_sha256'] for r in H if r['event'] in ('baseline','single','joint') and r.get('executed')}
continuation_hashes=set();calls_continuation=0
full_cache={};focal_cache={}
for r in H:
 if r['event']=='baseline':
  h=r['input_sha256'];origin=oldkey(r)
  if h in full_cache:assert full_cache[h][0]==r['q']
  else:full_cache[h]=(r['q'],origin,'v2')
  focus=r['focus_class'];focal_cache[h,focus]=(r['q'][focus],origin,'v2')
for r in H:
 if r['event'] in ('single','joint'):
  h=r['input_sha256'];focus=prior_baselines[r['flow_id']]['focus_class'];origin=oldkey(r)
  if (h,focus) in focal_cache:assert focal_cache[h,focus][0]==r['q_focus']
  else:focal_cache[h,focus]=(r['q_focus'],origin,'v2')
  if h in full_cache:assert full_cache[h][0][focus]==r['q_focus']
try:
 import psutil
 creation_time=psutil.Process().create_time()
except ImportError:creation_time=None
emit('execution_start',pid=os.getpid(),create_time=creation_time,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),protocol_sha256=P['protocol_sha256'],immutable_v2_prefix_sha256=P['immutable_v2_prefix_sha256'],immutable_v2_prefix_events=len(H),model=P['model'])
emit('execution_resume',completed_flow_ids=sorted(completed_flows),prior_event_keys=len(prior),prior_full_q_cache_entries=len(full_cache),prior_focal_cache_entries=len(focal_cache),historical_model_evaluations=historical_calls,historical_distinct_evaluated_hashes=len(historical_hashes),cache_policy='Baseline requests use full q only; focal requests may reuse exact (inputSHA,focusClass). Every newly scored input uses v2 full reset.')
assert digest(P['model'])==P['model_sha256'],'Checkpoint changed'
b=json.loads(Path('/home/aaup/Desktop/pi5_h3_measure/xai_pi_bundle.json').read_text())
assert hashlib.sha256(Path('/home/aaup/Desktop/pi5_h3_measure/xai_pi_bundle.json').read_bytes()).hexdigest()==P['bundle_sha256']
llm=Llama(model_path=P['model'],n_ctx=2048,n_threads=4,n_batch=512,logits_all=True,verbose=False)
pre=b['prefix_ids'];nodes=b['branch_nodes'];kids=b['children'];paths=b['paths'];classes=list(paths)
first=b['flows'][0]['ids'];user_start=first.index(32010)+1;user_end=first.index(32007,user_start)
prompt_prefix=first[:user_start];prompt_suffix=first[user_end:]
for archived in b['flows']:
 original=archived['ids'];start=original.index(32010)+1;end=original.index(32007,start)
 assert original[:start]==prompt_prefix and original[end:]==prompt_suffix
 body=llm.detokenize(original[start:end]).decode().lstrip(' ')
 assert prompt_prefix+llm.tokenize(body.encode(),add_bos=False,special=False)+prompt_suffix==original
emit('execution_tokenization_verified',retained_prompts=150,exact_id_matches=150,bos_added=False,protocol_sha256=P['protocol_sha256'])
order=P['field_order'];system=P['system_prompt']
def parse(block):
 return dict(x.split('=',1) for x in block.split('\n',1)[1].split('  '))
def render(values):
 return 'Flow features:\n'+'  '.join(k+'='+values[k] for k in order if k in values)
def intervene(values,donor,fields):
 out=dict(values)
 for k in fields:
  if k in donor:out[k]=donor[k]
  else:out.pop(k,None)
 return render(out)
def checks(values):
 issues=[]
 for k in ('tcp_srcport','tcp_dstport'):
  if k in values:
   try:
    v=float(values[k])
    if not math.isfinite(v) or not v.is_integer() or not 0<=v<=65535:issues.append(k+':invalid_port')
   except (ValueError,OverflowError):issues.append(k+':non_numeric_port')
 try:
  flag_value=float(values.get('tcp_flags','0'))
  ack_value=float(values.get('tcp_flags_ack','0'))
  if not math.isfinite(flag_value) or not flag_value.is_integer() or not 0<=flag_value<=511:issues.append('tcp_flags_invalid')
  elif not math.isfinite(ack_value) or ack_value not in (0,1):issues.append('tcp_ack_invalid')
  elif bool(int(flag_value)&16)!=bool(int(ack_value)):issues.append('tcp_ack_flag_disagreement_under_zero_omission')
 except (ValueError,OverflowError):issues.append('tcp_flag_not_numeric')
 return issues

def score(block,focus,current_event_key):
 global calls_continuation
 key=hashlib.sha256(block.encode()).hexdigest()
 if key in full_cache:
  q,origin,phase=full_cache[key]
  source='prior_baseline_full' if phase=='v2' else 'continuation_full'
  return (q if focus is None else q[focus]),False,key,source,origin,None
 if focus is not None and (key,focus) in focal_cache:
  value,origin,phase=focal_cache[key,focus]
  source='prior_focal_scalar' if phase=='v2' else 'continuation_focal_scalar'
  return value,False,key,source,origin,None
 # Strict v2 inference implementation: full reset and identical trie arithmetic.
 ids=prompt_prefix+llm.tokenize(block.encode(),add_bos=False,special=False)+prompt_suffix
 llm.reset();llm.eval(ids)
 probs={}
 for node in nodes:
  llm.n_tokens=len(ids);llm.eval(pre+node)
  allowed=kids[json.dumps(node)];allowed=[int(k) for k in allowed]
  z=np.array(llm.scores[llm.n_tokens-1][allowed],dtype=np.float64)
  ex=np.exp(z-z.max());weights=ex/ex.sum()
  probs[tuple(node)]={k:float(p) for k,p in zip(allowed,weights)}
 q={}
 for name,path in paths.items():
  value=1.
  for pos,token in enumerate(path):
   prefix=tuple(path[:pos])
   if prefix in probs:value*=probs[prefix][token]
  q[name]=value
 assert abs(sum(q.values())-1)<1e-10,(sum(q.values()),q)
 assert all(math.isfinite(x) and x>=0 for x in q.values())
 # Never silently change a recovered value if a previously scalar-only input is re-evaluated for full q.
 for (h,c),(old_value,origin,phase) in focal_cache.items():
  if h==key:assert q[c]==old_value,('Recovered focal value changed',key,c,old_value,q[c])
 calls_continuation+=1;continuation_hashes.add(key)
 full_cache[key]=(q,current_event_key,'v3')
 for c,value in q.items():focal_cache[key,c]=(value,current_event_key,'v3')
 return (q if focus is None else q[focus]),True,key,'full_reset_evaluation',current_event_key,q

def emit_metric(event,flow_id,**kw):
 key=kw['event_key'];assert key not in logged_keys,('Duplicate event',key)
 emit(event,flow_id=flow_id,**kw);logged_keys.add(key)
def cachekw(source,origin,qvec):
 out={'cache_source':source,'cache_source_event_key':origin}
 if qvec is not None:out['q_vector']=qvec
 return out
def counters():
 return {'historical_model_evaluations':historical_calls,'continuation_model_evaluations':calls_continuation,'actual_model_evaluations_total':historical_calls+calls_continuation,'distinct_input_hashes_evaluated_total':len(historical_hashes|continuation_hashes)}

donors=[parse(d['flow_block']) for d in P['donors']]
t0=time.monotonic()
emit('execution_model_loaded',protocol_sha256=P['protocol_sha256'],model_sha256=P['model_sha256'],target_flows=len(P['flows']),donors=len(donors),hardware=P['hardware'],n_ctx=2048,n_threads=4,n_batch=512,logits_all=True,prefix_reuse=False)
for fi,flow in enumerate(P['flows']):
 if flow['i'] in completed_flows:
  emit('execution_flow_reused',flow_id=flow['i'],ordinal=fi+1,total=len(P['flows']),prior_flow_complete_event_key=eventkey('flow_complete',flow['i']))
  continue
 block=flow['flow_block'];values=parse(block);fields=[k for k in order if k in values]
 key=eventkey('baseline',flow['i'])
 if key in prior:
  saved=prior[key];q0=saved['q'];focus=saved['focus_class']
  assert saved['input_sha256']==hashlib.sha256(block.encode()).hexdigest()
 else:
  q0,fresh,inputsha,source,origin,qvec=score(block,None,key);focus=max(q0,key=q0.get)
  emit_metric('baseline',flow['i'],event_key=key,source_row=flow['source_row'],true=flow['true'],focus_class=focus,q=q0,input_sha256=inputsha,input_block=block,original_issues=checks(values),executed=fresh,**cachekw(source,origin,None))
 log0=math.log(max(q0[focus],1e-12));drops={}
 for di,donor in enumerate(donors):
  for field in fields:
   key=eventkey('single',flow['i'],di,field)
   altered=intervene(values,donor,[field])
   if key in prior:
    saved=prior[key];assert saved['input_sha256']==hashlib.sha256(altered.encode()).hexdigest()
    drops[di,field]=saved['drop_log_probability']
    continue
   value,fresh,inputsha,source,origin,qvec=score(altered,focus,key)
   drop=log0-math.log(max(value,1e-12));drops[di,field]=drop
   emit_metric('single',flow['i'],event_key=key,donor_id=P['donors'][di]['donor_id'],donor_index=di,field=field,drop_log_probability=drop,q_focus=value,input_sha256=inputsha,unchanged=(altered==block),executed=fresh,issues=checks(parse(altered)),**cachekw(source,origin,qvec))
   if fresh and calls_continuation%10==0:emit('execution_progress',flow_id=flow['i'],ordinal=fi+1,total=len(P['flows']),completed_flow_count=len(completed_flows),elapsed_s=round(time.monotonic()-t0,3),elapsed_scope='continuation',**counters())
 k=min(3,len(fields));rng=np.random.default_rng(P['random_control_seed']+flow['i'])
 random_fields=list(rng.choice(fields,size=k,replace=False))
 for di,donor in enumerate(donors):
  scores={field:float(np.mean([drops[d,field] for d in range(len(donors)) if d!=di])) for field in fields}
  selected=sorted(fields,key=lambda f:(-scores[f],order.index(f)))[:k]
  individual=sorted(fields,key=lambda f:(-drops[di,f],order.index(f)))[:k]
  for label,selection in (('leave_one_donor_out',selected),('random',random_fields)):
   key=eventkey('joint',flow['i'],di,label);altered=intervene(values,donor,selection)
   if key in prior:
    saved=prior[key];assert saved['selected_fields']==selection and saved['input_sha256']==hashlib.sha256(altered.encode()).hexdigest()
    continue
   value,fresh,inputsha,source,origin,qvec=score(altered,focus,key)
   emit_metric('joint',flow['i'],event_key=key,donor_index=di,donor_id=P['donors'][di]['donor_id'],method=label,selected_fields=selection,single_donor_top3=individual,drop_log_probability=log0-math.log(max(value,1e-12)),q_focus=value,input_sha256=inputsha,executed=fresh,issues=checks(parse(altered)),**cachekw(source,origin,qvec))
 key=eventkey('flow_complete',flow['i'])
 completed_flows.add(flow['i'])
 emit_metric('flow_complete',flow['i'],event_key=key,ordinal=fi+1,total=len(P['flows']),present_fields=len(fields),distinct_model_evaluations=len(historical_hashes|continuation_hashes),elapsed_s=round(time.monotonic()-t0,3),elapsed_scope='continuation',thermal_c=int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000,**counters())
assert len(completed_flows)==len(P['flows'])
emit('complete',flows=len(P['flows']),protocol_sha256=P['protocol_sha256'],immutable_v2_prefix_sha256=P['immutable_v2_prefix_sha256'],immutable_v2_prefix_events=len(H),distinct_model_evaluations=len(historical_hashes|continuation_hashes),elapsed_s=round(time.monotonic()-t0,3),elapsed_scope='continuation',prefix_reuse=False,**counters())
