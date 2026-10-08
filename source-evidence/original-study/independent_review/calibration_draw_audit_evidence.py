"""Read original CSV/JSON/NPZ evidence only; never import original experiment code."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, math, sys, itertools
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
SOURCE = Path(r'F:\server_xai\FiveCapterThis')
XAI = SOURCE / 'xai2026'
DEST = Path(r'C:\Users\MJO\Downloads\New folder (9)\Original_XAI_recovery_20261008')
DEST.mkdir(parents=True, exist_ok=True)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def load_lines(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8-sig').splitlines() if l.strip()]

def assert_record(ok,message):
    if not bool(ok): raise AssertionError(message)

cfg=load_json(SOURCE/'transfer/bundle/config.json')
classes=cfg['classes']; ci={c:i for i,c in enumerate(classes)}
manifest=load_json(XAI/'out/draws/manifest.json')
hashes={}
def record_hash(p):
    p=Path(p); hashes[str(p)]={'sha256':digest(p),'bytes':p.stat().st_size}; return hashes[str(p)]['sha256']

indices={}
for name,m in manifest.items():
    p=XAI/'out/draws'/f'{name}_index.csv'; d=pd.read_csv(p)
    sha=record_hash(p)
    assert_record(sha==m['index_sha256'],name+' index hash')
    assert_record(len(d)==m['n'] and len(d.row.unique())==len(d),name+' index size/unique')
    assert_record(d.Attack_type.value_counts().to_dict()==m['per_class'],name+' index labels')
    indices[name]=d

val_csv=SOURCE/'transfer/data/splits/val.csv'
test_csv=SOURCE/'transfer/bundle/artifacts/samples/largesample_n6184.csv'
real_csv=SOURCE/'transfer/bundle/artifacts/samples/realistic_n600.csv'
cal_csv=XAI/'out/draws/val_cal.csv'
fields=[c for _,c,_ in cfg['flow_block']['field_order']]
val=pd.read_csv(val_csv,usecols=fields+['Attack_type'],low_memory=False)
cal=pd.read_csv(cal_csv,index_col=0,low_memory=False)
test=pd.read_csv(test_csv,low_memory=False); real=pd.read_csv(real_csv,low_memory=False)
source_val_n=len(val)

def stratified(df,cap):
    return pd.concat([df[df.Attack_type.astype(str)==c].sample(min(cap,int((df.Attack_type.astype(str)==c).sum())),random_state=2026) for c in classes if (df.Attack_type.astype(str)==c).any()]).sort_index()

recon={'val_cal':stratified(val,250),'val_sub1500':stratified(cal,100),'test_sub1500':stratified(test,100)}
recon['val_sub300']=stratified(recon['val_sub1500'],20)
recon['test_sub300']=stratified(recon['test_sub1500'],20)
for name,df in recon.items():
    expected=indices[name]
    assert_record(df.index.tolist()==expected.row.tolist(),name+' reproduced indices')
    assert_record(df.Attack_type.astype(str).tolist()==expected.Attack_type.tolist(),name+' reproduced labels')
assert_record(cal.index.tolist()==indices['val_cal'].row.tolist(),'retained calibration CSV indices')
assert_record(cal.Attack_type.tolist()==indices['val_cal'].Attack_type.tolist(),'retained calibration CSV labels')

def block(row):
    parts=[]
    for label,column,rule in cfg['flow_block']['field_order']:
        if column not in row.index: continue
        v=row[column]
        if v is None or (isinstance(v,float) and math.isnan(v)): continue
        s=str(v).strip()
        if not s or s.lower()=='nan': continue
        if rule=='nonzero':
            try:
                if float(s)==0.0: continue
            except ValueError: pass
        parts.append(label+'='+s)
    return cfg['flow_block']['header']+'\n'+cfg['flow_block']['separator'].join(parts)

cal_blocks={int(i):block(row) for i,row in cal.iterrows()}
source_blocks={int(i):block(row) for i,row in recon['val_cal'].iterrows()}
cal_source_block_mismatch=[i for i,b in cal_blocks.items() if b!=source_blocks[i]]
assert_record(not cal_source_block_mismatch,'calibration render agrees with selected original validation rows')
test_blocks={int(i):block(row) for i,row in test.iterrows()}
real_blocks={int(i):block(row) for i,row in real.iterrows()}

frames={'val_cal':cal,'val_sub1500':cal.loc[indices['val_sub1500'].row],'val_sub300':cal.loc[indices['val_sub300'].row],
        'test':test,'test_sub1500':test.loc[indices['test_sub1500'].row],'test_sub300':test.loc[indices['test_sub300'].row],'realistic':real}
logs={}; bindings={}; arrays={}
for p in sorted((XAI/'out').glob('x1_*.jsonl')):
    tag=p.stem.removeprefix('x1_'); model=next((m for m in ['q8_s456','q8_s42','q8_s123','q4_s456','bf16_s456'] if tag.startswith(m+'_')),None)
    if model is None:continue
    tail=tag[len(model)+1:]; draw,mode=tail.rsplit('_',1)
    if draw not in frames:continue
    rows=load_lines(p); expected=indices[draw]
    assert_record(len(rows)==len(expected),p.name+' record count')
    assert_record(len({r['row'] for r in rows})==len(rows),p.name+' unique keys')
    assert_record([r['row'] for r in rows]==expected.row.tolist(),p.name+' record ordering')
    df=frames[draw]
    mismatches=[]
    for r in rows:
        i=r['row']; rr=df.loc[i]
        if r['key']!=i or r['true']!=str(rr.Attack_type) or r['flow_block']!=block(rr):mismatches.append(i)
    assert_record(not mismatches,p.name+' row/truth/block binding')
    record_hash(p); logs[(model,draw,mode)]=rows
    bindings[p.name]={'n':len(rows),'draw':draw,'model':model,'mode':mode,'unique_row_keys':len({r['row'] for r in rows}),'manifest_order_truth_and_rendered_block_match':True,'parsed_count':sum(bool(r.get('parsed')) for r in rows),'missing_trie_tokens':sum(r.get('missing',0) for r in rows)}

for p in sorted((XAI/'out').glob('x2_*_scores.npz')):
    tag=p.stem.removeprefix('x2_').removesuffix('_scores')
    model=next((m for m in ['q8_s456','q8_s42','q8_s123','q4_s456','bf16_s456'] if tag.startswith(m+'_')),None)
    if not model:continue
    draw=tag[len(model)+1:]; rows=logs[(model,draw,'dec')]
    with np.load(p,allow_pickle=False) as a:
        y=np.array([ci[r['true']] for r in rows]); lab=np.array([ci.get(r['label'],-1) for r in rows]); correct=(y==lab).astype(int)
        assert_record(np.array_equal(a['y'],y) and np.array_equal(a['lab'],lab) and np.array_equal(a['correct'],correct),p.name+' labels/correct binding')
        P=np.array([[r['dist'][c] for c in classes] for r in rows]); raw=P[np.arange(len(rows)),np.clip(lab,0,len(classes)-1)]
        assert_record(np.allclose(a['s_intrinsic_raw'],raw,atol=1e-14,rtol=0),p.name+' intrinsic score binding')
        assert_record(np.allclose(a['s_verbal_raw'],[r['conf'] or 0.0 for r in rows],atol=1e-14,rtol=0),p.name+' verbal score binding')
        arrays[p.name]={'n':len(y),'keys':list(a.keys()),'shapes':{k:list(a[k].shape) for k in a.keys()},'label_correct_and_raw_score_binding':True,'set_coverage':float(a['sets'][np.arange(len(y)),y].mean()),'mean_set_size':float(a['sets'].sum(1).mean())}
    record_hash(p)

def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);return reader.fieldnames,list(reader)

test_fields,test_raw=csv_rows(test_csv); real_fields,real_raw=csv_rows(real_csv)
selected=set(cal.index.tolist()); val_raw={}; val_labels=[]; val_fields=None
with val_csv.open(encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f); val_fields=reader.fieldnames
    for i,r in enumerate(reader):
        val_labels.append(r['Attack_type'])
        if i in selected:val_raw[i]=r
assert_record(len(val_labels)==len(val) and len(val_raw)==3582,'original validation raw selection')
nonlabels=sorted(set(test_fields)-{'Attack_type','Attack_label'})
assert_record(nonlabels==sorted(set(real_fields)-{'Attack_type','Attack_label'})==sorted(set(val_fields)-{'Attack_type','Attack_label'}),'identical nonlabel source schema')
def source_hash(r):
    b=json.dumps([[c,r[c]] for c in nonlabels],ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return hashlib.sha256(b).hexdigest()

raws={'val_cal':val_raw,'test':dict(enumerate(test_raw)),'realistic':dict(enumerate(real_raw)),'test_sub1500':{int(i):test_raw[int(i)] for i in indices['test_sub1500'].row}}
rendered={'val_cal':cal_blocks,'test':test_blocks,'realistic':real_blocks,'test_sub1500':{int(i):test_blocks[int(i)] for i in indices['test_sub1500'].row}}
source_hashes={n:{i:source_hash(r) for i,r in rs.items()} for n,rs in raws.items()}
def overlap(a,b,kind):
    ah=source_hashes[a] if kind=='source' else rendered[a];bh=source_hashes[b] if kind=='source' else rendered[b]
    aset=set(ah.values());bset=set(bh.values())
    ia=[i for i,v in ah.items() if v in bset];ib=[i for i,v in bh.items() if v in aset]
    shared=aset&bset
    return {'shared_unique':len(shared),'left_rows_matching':len(ia),'right_rows_matching':len(ib),'left_row_ids':ia,'right_row_ids':ib,'left_labels':dict(Counter(raws[a][i]['Attack_type'] for i in ia)),'right_labels':dict(Counter(raws[b][i]['Attack_type'] for i in ib))}

overlaps={}
for a,b in itertools.combinations(raws,2):
    overlaps[a+'__'+b]={'nonlabel_source_csv_string':overlap(a,b,'source'),'rendered_input':overlap(a,b,'rendered')}
triple=set(source_hashes['test'].values())&set(source_hashes['realistic'].values())&set(source_hashes['test_sub1500'].values())
overlaps['test__realistic__test_sub1500']={'shared_unique_nonlabel_source_csv_strings':len(triple)}

fv=logs[('q8_s456','val_cal','dec')]; corr=np.array([int(r['label']==r['true']) for r in fv])
cv=[]
# Independent reproduction of StratifiedKFold's round-robin class allocation
# and MT19937 shuffle, without importing the unavailable sklearn dependency.
# These memberships are audit derivatives, not originally retained fold records.
_, y_idx, y_inv = np.unique(corr, return_index=True, return_inverse=True)
_, class_perm = np.unique(y_idx, return_inverse=True)
y_encoded=class_perm[y_inv]
y_order=np.sort(y_encoded)
allocation=np.asarray([np.bincount(y_order[i::5],minlength=2) for i in range(5)])
rng=np.random.RandomState(0); fold_ids=np.empty(len(corr),dtype=int)
for k in range(2):
    folds_for_class=np.arange(5).repeat(allocation[:,k]);rng.shuffle(folds_for_class)
    fold_ids[y_encoded==k]=folds_for_class
for fold in range(5):
    te=np.flatnonzero(fold_ids==fold);tr=np.flatnonzero(fold_ids!=fold)
    cv.append({'fold':fold,'fit_n':len(tr),'heldout_n':len(te),'fit_correct':int(corr[tr].sum()),'heldout_correct':int(corr[te].sum()),'fit_row_ids':[fv[int(i)]['row'] for i in tr],'heldout_row_ids':[fv[int(i)]['row'] for i in te],'row_positions_disjoint':not bool(set(tr)&set(te)),'heldout_blocks_seen_in_fold_fit':sum(fv[int(i)]['flow_block'] in {fv[int(j)]['flow_block'] for j in tr} for i in te)})
primary=load_json(XAI/'out/x2_trackA_q8_s456_val_cal.json')
assert_record(primary['n_val']==len(fv),'trackA n_val')
assert_record(primary['primary_calibrator']==min(primary['cv'],key=lambda k:primary['cv'][k]['ece']),'CV calibration winner')
assert_record(primary['primary_ranking']==max(primary['cv'],key=lambda k:primary['cv'][k]['auroc']),'CV ranking winner')
for p in [XAI/'xai_common.py',XAI/'x0_gates.py',XAI/'x1_generate.py',XAI/'x2_trackA.py',XAI/'out/draws/manifest.json',XAI/'out/x2_trackA_q8_s456_val_cal.json',SOURCE/'transfer/bundle/config.json',SOURCE/'transfer/bundle/secedge_lib.py',val_csv,test_csv,real_csv,cal_csv]:record_hash(p)

membership=DEST/'original_calibration_membership_evidence_audit.csv'
with membership.open('w',encoding='utf-8',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=['validation_csv_offset','label','nonlabel_source_csv_string_sha256','rendered_flow_block_sha256','prediction','correct']);wr.writeheader()
    for r in fv:
        i=r['row'];wr.writerow({'validation_csv_offset':i,'label':r['true'],'nonlabel_source_csv_string_sha256':source_hashes['val_cal'][i],'rendered_flow_block_sha256':hashlib.sha256(r['flow_block'].encode('utf-8')).hexdigest(),'prediction':r['label'],'correct':int(r['label']==r['true'])})

audit={
 'status':'VERIFIED_ORIGINAL_MEMBERSHIP_AND_DRAW_BINDINGS_WITH_INDEPENDENCE_LIMITATIONS',
 'source_root':str(XAI),'method':'Independent data-only Python audit. Original scripts read as text, never imported or executed; no model inference or pickle deserialization.',
 'source_hashes':hashes,
 'source_partitions':{'validation':source_val_n,'training':'1,553,434 declared in earlier authenticated source-pool evidence; full training pool not rescanned by this audit','test_pool':'332,893 in earlier authenticated source-pool evidence; actual XAI frozen sample verified here'},
 'draws':{n:{**m,'original_index_min':int(indices[n].row.min()),'original_index_max':int(indices[n].row.max()),'original_index_space':'validation CSV offset' if n.startswith('val') else ('6,184-row frozen sample offset' if n.startswith('test') else '600-row realistic sample offset')} for n,m in manifest.items()},
 'draw_reconstruction':{'seed':2026,'original_calibration_rows_reproduced_from_validation':True,'calibration_source_render_matches_saved_csv_and_logs':True,'nested_test_sub1500_reproduced':True,'nested_test_sub300_reproduced':True,'nested_val_sub1500_reproduced':True,'nested_val_sub300_reproduced':True,'test_and_realistic_source_CSV_sha_verified':True},
 'log_bindings':bindings,'score_array_bindings':arrays,
 'primary_calibration':{'n':3582,'correct':int(corr.sum()),'incorrect':int(len(corr)-corr.sum()),'unique_rendered_inputs':len(set(cal_blocks.values())),'selection':'5-fold StratifiedKFold by prediction correctness, shuffle=True, seed0, validation only','cv_fold_provenance':'Audit derivative recreating round-robin stratified class allocation and MT19937 shuffle. Original per-fold membership records are not retained; no calibrator fits were rerun.','cv_folds':cv,'primary_calibrator':primary['primary_calibrator'],'primary_ranking':primary['primary_ranking'],'temperature':primary['temperature'],'final_fit_rows':3582,'conformal_quantile_rows':3582,'temperature_and_conformal_quantile_same_rows':True,'separate_temperature_fit_conformal_calibration_partition':False,'deferral_threshold_selection':'same validation rows, validation-error-score 0.406 quantile','test_labels_used_by_calibration_selection_code':False},
 'overlap_conventions':{'nonlabel_source_csv_string':'SHA256 of UTF8 compact JSON sorted [column,original CSV string] pairs; Attack_type and Attack_label excluded. This measures exact retained source-value equality, not physical-flow identity.','rendered_input':'Exact flow_block string as in retained logs, reconstructed independently using config14fields/presence rules. Different truth labels can share a block.','index_space_warning':'test/realistic integer keys use different sample-relative offset spaces; do not intersect integer keys directly. Validation offsets refer to original validation CSV.'},
 'overlaps':overlaps,
 'limitations':['Temperature was fitted on the same 3,582 rows used for conformal quantiles; ordinary split-conformal finite-sample coverage is not independently authenticated by this procedure.','Cross-draw exact source-value and rendered-input overlap must be reported; distinct source partitions do not establish feature/input independence.','No explicit train/test duplicate or rendered-input exclusion is present in original draw construction. Fitted detector/checkpoint training membership is not established by this audit; existing training-pool overlap evidence remains separate.','Five-fold CV is disjoint by row position, not grouped by rendered input. Duplicate rendered blocks can occur across fit/heldout folds.','This authenticates retained artifacts and output-to-draw bindings; it does not independently establish every historical execution timestamp or physical-flow independence.'],
 'derived_membership_file':{'path':str(membership),'sha256':digest(membership),'n':3582}
}
out=DEST/'calibration_draw_provenance_evidence_audit.json';out.write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'report':str(out),'sha256':digest(out),'calibration_n':3582,'unique_calibration_blocks':len(set(cal_blocks.values())),'all_log_n':{n:b['n'] for n,b in bindings.items()},'overlaps':{pair:{kind:{k:v for k,v in obj.items() if k not in ['left_row_ids','right_row_ids']} if isinstance(obj,dict) else obj for kind,obj in vals.items()} for pair,vals in overlaps.items()},'cv_fold_sizes':[(f['fit_n'],f['heldout_n'],f['heldout_blocks_seen_in_fold_fit']) for f in cv]},indent=2))
