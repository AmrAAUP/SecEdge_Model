"""Independent streaming source-pool overlap; no original imports, fits or models."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, math, sys, time
sys.stdout.reconfigure(encoding='utf-8')
R=Path(r'F:\server_xai\FiveCapterThis'); X=R/'xai2026'
D=Path(r'C:\Users\MJO\Downloads\New folder (9)\Original_XAI_recovery_20261008')
cfg=json.loads((R/'transfer/bundle/config.json').read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def raw_block(row):
 out=[]
 for label,col,rule in cfg['flow_block']['field_order']:
  t=row.get(col,'').strip()
  if not t or t.lower()=='nan':continue
  if rule=='nonzero':
   try:
    if float(t)==0.0:continue
   except ValueError:pass
  out.append(label+'='+t)
 return cfg['flow_block']['header']+'\n'+cfg['flow_block']['separator'].join(out)

q={}
cal_members=list(csv.DictReader((D/'original_calibration_membership_evidence_audit.csv').open(encoding='utf-8',newline='')))
q['calibration']={int(r['validation_csv_offset']):{'source':r['nonlabel_source_csv_string_sha256'],'block':r['rendered_flow_block_sha256'],'label':r['label']} for r in cal_members}
schema=None
for name,csvpath,logname in [('test',R/'transfer/bundle/artifacts/samples/largesample_n6184.csv','x1_q8_s456_test_dec.jsonl'),('realistic',R/'transfer/bundle/artifacts/samples/realistic_n600.csv','x1_q8_s456_realistic_dec.jsonl')]:
 log=lines(X/'out'/logname);q[name]={}
 with csvpath.open(encoding='utf-8-sig',newline='') as f:
  rr=csv.DictReader(f);s=sorted(set(rr.fieldnames)-{'Attack_type','Attack_label'})
  if schema is None:schema=s
  assert schema==s
  for i,row in enumerate(rr):
   sh=hashlib.sha256(json.dumps([[c,row[c]] for c in schema],ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()
   b=raw_block(row);assert b==log[i]['flow_block'],(name,i,'raw renderer differs from original log')
   q[name][i]={'source':sh,'block':hashlib.sha256(b.encode()).hexdigest(),'label':row['Attack_type']}
q['explanation1500']={r['row']:q['test'][r['row']] for r in lines(X/'out/x1_q8_s456_test_sub1500_full.jsonl')}
query={kind:set(z[kind] for group in q.values() for z in group.values()) for kind in ['source','block']}
hits={kind:defaultdict(lambda:{'n_training_rows':0,'training_labels':Counter()}) for kind in query}
path=R/'transfer/data/splits/train.csv';h=hashlib.sha256(); t0=time.monotonic()
with path.open(encoding='utf-8-sig',newline='') as f:
 rr=csv.DictReader(f);assert sorted(set(rr.fieldnames)-{'Attack_type','Attack_label'})==schema
 for n,row in enumerate(rr,1):
  sh=hashlib.sha256(json.dumps([[c,row[c]] for c in schema],ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()
  bh=hashlib.sha256(raw_block(row).encode('utf-8')).hexdigest()
  for kind,key in [('source',sh),('block',bh)]:
   if key in query[kind]:
    hits[kind][key]['n_training_rows']+=1;hits[kind][key]['training_labels'][row['Attack_type']]+=1
  if n%250000==0:print(json.dumps({'training_rows_scanned':n,'elapsed_seconds':round(time.monotonic()-t0,1)}),flush=True)

out={'status':'VERIFIED_FULL_TRAINING_POOL_SOURCE_VALUE_AND_RENDERED_INPUT_OVERLAP','training_csv':str(path),'training_n':n,'training_csv_sha256':digest(path),'runtime_seconds':time.monotonic()-t0,'conventions':{'source':'SHA256 compact UTF8 JSON sorted non-label [column,original CSV string] pairs; Attack_type/Attack_label excluded. Exact value equality does not identify physical flows.','block':'SHA256 exact config14-field rendered flow block; raw CSV renderer validated against every original test and realistic log block.','scope':'Declared full training pool; not checkpoint consumed-example or fitted-gate membership.'},'draws':{}}
for name,group in q.items():
 out['draws'][name]={'n':len(group)}
 for kind in query:
  matched=[i for i,z in group.items() if z[kind] in hits[kind]]
  uniq={group[i][kind] for i in matched}
  out['draws'][name][kind]={'query_occurrences_matching_training':len(matched),'query_matching_row_ids':matched,'query_matching_labels':dict(Counter(group[i]['label'] for i in matched)),'shared_unique_hashes':len(uniq),'training_occurrences_matching_this_draw':sum(hits[kind][k]['n_training_rows'] for k in uniq),'matched_hashes_with_other_training_labels':sum(any(l!=group[i]['label'] for l in hits[kind][group[i][kind]]['training_labels']) for i in matched),'matching_query_ids_are_sample_offsets_except_calibration_original_validation_offsets':True}
out['limitations']=['No duplicate/source-value/rendered-input removal is present in the original XAI draw-building code.','This measures full-pool overlap; the exact historical detector and checkpoint fitting membership is not authenticated.','No records or outcomes were excluded retrospectively.']
p=D/'training_overlap_provenance_evidence_audit.json';p.write_text(json.dumps(out,indent=2),encoding='utf-8')
brief={name:{k:v for k,v in d.items() if k=='n' or isinstance(v,dict)} for name,d in out['draws'].items()}
for name,d in brief.items():
 for k,v in d.items():
  if isinstance(v,dict):v.pop('query_matching_row_ids',None)
print(json.dumps({'report':str(p),'sha256':digest(p),'training_n':n,'training_sha256':out['training_csv_sha256'],'counts':brief},indent=2))
