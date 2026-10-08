import os,json,time,hashlib,math,itertools
from pathlib import Path
import numpy as np
from llama_cpp import Llama
def emit(event,**kw):
 print(json.dumps(dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)),flush=True)
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
model=P['model']
emit('starting',pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),protocol_sha256=P['protocol_sha256'],model=model)
assert digest(model)==P['model_sha256'],'Checkpoint changed'
b=json.loads(Path('/home/aaup/Desktop/pi5_h3_measure/xai_pi_bundle.json').read_text())
assert hashlib.sha256(Path('/home/aaup/Desktop/pi5_h3_measure/xai_pi_bundle.json').read_bytes()).hexdigest()==P['bundle_sha256']
llm=Llama(model_path=model,n_ctx=2048,n_threads=4,n_batch=512,logits_all=True,verbose=False)
pre=b['prefix_ids']; nodes=b['branch_nodes']; kids=b['children']; paths=b['paths']; classes=list(paths)
first=b['flows'][0]['ids']; user_start=first.index(32010)+1; user_end=first.index(32007,user_start)
prompt_prefix=first[:user_start];prompt_suffix=first[user_end:]
for archived in b['flows']:
 original=archived['ids']; start=original.index(32010)+1; end=original.index(32007,start)
 assert original[:start]==prompt_prefix and original[end:]==prompt_suffix
 body=llm.detokenize(original[start:end]).decode().lstrip(' ')
 assert prompt_prefix+llm.tokenize(body.encode(),add_bos=False,special=False)+prompt_suffix==original
emit('tokenization_verified',retained_prompts=150,exact_id_matches=150,bos_added=False)
order=P['field_order']; system=P['system_prompt']
cache={}; calls=0
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
def score(block):
 global calls
 key=hashlib.sha256(block.encode()).hexdigest()
 if key in cache:return cache[key],False,key
 ids=prompt_prefix+llm.tokenize(block.encode(),add_bos=False,special=False)+prompt_suffix
 llm.reset(); llm.eval(ids)
 probs={}
 for node in nodes:
  llm.n_tokens=len(ids);llm.eval(pre+node)
  allowed=kids[json.dumps(node)]
  allowed=[int(k) for k in allowed]
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
 calls+=1;cache[key]=q
 return q,True,key
donors=[parse(d['flow_block']) for d in P['donors']]
t0=time.monotonic()
emit('model_loaded',model_sha256=P['model_sha256'],target_flows=len(P['flows']),donors=len(donors),hardware=P['hardware'])
for fi,flow in enumerate(P['flows']):
 block=flow['flow_block'];values=parse(block);fields=[k for k in order if k in values]
 q0,fresh,key=score(block);focus=max(q0,key=q0.get);log0=math.log(max(q0[focus],1e-12))
 emit('baseline',flow_id=flow['i'],source_row=flow['source_row'],true=flow['true'],focus_class=focus,q=q0,input_sha256=key,input_block=block,original_issues=checks(values),executed=fresh)
 drops={}
 for di,donor in enumerate(donors):
  for field in fields:
   altered=intervene(values,donor,[field]);q,executed,key=score(altered)
   drop=log0-math.log(max(q[focus],1e-12));drops[di,field]=drop
   emit('single',flow_id=flow['i'],donor_id=P['donors'][di]['donor_id'],donor_index=di,field=field,drop_log_probability=drop,q_focus=q[focus],input_sha256=key,unchanged=(altered==block),executed=executed,issues=checks(parse(altered)))
 k=min(3,len(fields));rng=np.random.default_rng(P['random_control_seed']+flow['i'])
 random_fields=list(rng.choice(fields,size=k,replace=False))
 for di,donor in enumerate(donors):
  scores={field:float(np.mean([drops[d,field] for d in range(len(donors)) if d!=di])) for field in fields}
  selected=sorted(fields,key=lambda f:(-scores[f],order.index(f)))[:k]
  individual=sorted(fields,key=lambda f:(-drops[di,f],order.index(f)))[:k]
  for label,selection in (('leave_one_donor_out',selected),('random',random_fields)):
   altered=intervene(values,donor,selection);q,executed,key=score(altered)
   emit('joint',flow_id=flow['i'],donor_index=di,donor_id=P['donors'][di]['donor_id'],method=label,selected_fields=selection,single_donor_top3=individual,drop_log_probability=log0-math.log(max(q[focus],1e-12)),q_focus=q[focus],input_sha256=key,executed=executed,issues=checks(parse(altered)))
 emit('flow_complete',flow_id=flow['i'],ordinal=fi+1,total=len(P['flows']),present_fields=len(fields),distinct_model_evaluations=calls,elapsed_s=round(time.monotonic()-t0,3),thermal_c=int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000)
emit('complete',flows=len(P['flows']),distinct_model_evaluations=calls,elapsed_s=round(time.monotonic()-t0,3))

