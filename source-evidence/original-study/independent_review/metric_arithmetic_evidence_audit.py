"""Verify retained score-array arithmetic and simple validation CV, without models."""
from pathlib import Path
import json, hashlib, sys, math
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
X=Path(r'F:\server_xai\FiveCapterThis\xai2026')
D=Path(r'C:\Users\MJO\Downloads\New folder (9)\Original_XAI_recovery_20261008')
audit=json.loads((D/'calibration_draw_provenance_evidence_audit.json').read_text(encoding='utf-8'))
cfg=json.loads((X.parent/'transfer/bundle/config.json').read_text(encoding='utf-8-sig'))
classes=cfg['classes'];ci={c:i for i,c in enumerate(classes)}
def j(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ece(s,c,mass=False):
 if mass:groups=np.array_split(np.argsort(s),15)
 else:
  idx=np.clip(np.digitize(s,np.linspace(0,1,16)[1:-1],right=False),0,14)
  groups=[np.flatnonzero(idx==b) for b in range(15)]
 diffs=[(len(g),abs(float(s[g].mean()-c[g].mean()))) for g in groups if len(g)]
 return sum(n*v/len(s) for n,v in diffs),max(v for n,v in diffs)
def auc(s,c):
 order=np.argsort(s,kind='stable');ss=s[order];ranks=np.empty(len(s),float)
 starts=np.flatnonzero(np.r_[True,ss[1:]!=ss[:-1]])
 for st,en in zip(starts,np.r_[starts[1:],len(s)]):ranks[order[st:en]]=(st+1+en)/2
 pos=int(c.sum());neg=len(c)-pos
 return float((ranks[c==1].sum()-pos*(pos+1)/2)/(pos*neg))
def metrics(s,c):
 s=np.asarray(s,float);c=np.asarray(c,int);order=np.argsort(-s,kind='stable');err=1-c
 a=float(np.mean(np.cumsum(err[order])/np.arange(1,len(c)+1)))
 k=int(round((1-err.mean())*len(c)));oracle=np.r_[np.zeros(k),np.ones(len(c)-k)]
 out={'ece':ece(s,c)[0],'mce':ece(s,c)[1],'ece_mass':ece(s,c,True)[0],'auroc':auc(s,c),'aurc':a,'eaurc':a-float(np.mean(np.cumsum(oracle)/np.arange(1,len(c)+1))),'brier':float(np.mean((s-c)**2)),'nll':float(-np.mean(c*np.log(np.clip(s,1e-6,1))+(1-c)*np.log(np.clip(1-s,1e-6,1))))}
 for cov in [.8,.9,.95]:out['selacc_'+str(int(cov*100))]=float(c[order[:max(1,int(round(cov*len(c))))]].mean())
 return out
def hist_fit(v,c):
 edges=np.unique(np.quantile(v,np.linspace(0,1,16)))
 ix=np.clip(np.searchsorted(edges,v,side='right')-1,0,len(edges)-2)
 acc=np.array([c[ix==b].mean() if np.any(ix==b) else c.mean() for b in range(len(edges)-1)])
 return edges,acc
def hist_predict(v,fit):
 edges,acc=fit
 return acc[np.clip(np.searchsorted(edges,v,side='right')-1,0,len(edges)-2)]
cal=lines(X/'out/x1_q8_s456_val_cal_dec.jsonl');v=np.array([r['conf'] or 0.0 for r in cal]);c=np.array([int(r['label']==r['true']) for r in cal]);P=np.array([[r['dist'][cl] for cl in classes] for r in cal]);lab=np.array([ci.get(r['label'],-1) for r in cal]);raw=P[np.arange(len(cal)),np.clip(lab,0,14)];lookup={r['row']:i for i,r in enumerate(cal)}
cv={k:[] for k in ['verbal_raw','verbal_hist','intrinsic_raw']}
for fold in audit['primary_calibration']['cv_folds']:
 tr=np.array([lookup[i] for i in fold['fit_row_ids']]);te=np.array([lookup[i] for i in fold['heldout_row_ids']])
 for name,s in [('verbal_raw',v[te]),('verbal_hist',hist_predict(v[te],hist_fit(v[tr],c[tr]))),('intrinsic_raw',raw[te])]:
  mm=metrics(s,c[te]);cv[name].append({k:mm[k] for k in ['ece','auroc','aurc']})
rep=j(X/'out/x2_trackA_q8_s456_val_cal.json');cv_means={kind:{m:float(np.mean([r[m] for r in vals])) for m in ['ece','auroc','aurc']} for kind,vals in cv.items()}
cv_diffs={kind:{m:abs(cv_means[kind][m]-rep['cv'][kind][m]) for m in cv_means[kind]} for kind in cv_means}
assert max(d for vals in cv_diffs.values() for d in vals.values())<1e-12
records={}; maxdiff=0.0; maxmassdiff=0.0
for path in sorted((X/'out').glob('x2_*_scores.npz')):
 tag=path.stem[3:-7];model=next(m for m in ['q8_s456','q8_s42','q8_s123','q4_s456','bf16_s456'] if tag.startswith(m+'_'));draw=tag[len(model)+1:]
 valname='val_cal' if draw in ['test','realistic'] else ('val_sub300' if draw=='test_sub300' else 'val_sub1500')
 rr=j(X/'out'/f'x2_trackA_{model}_{valname}.json');expected=rr['tests'][draw]
 vals=lines(X/'out'/f'x1_{model}_{valname}_dec.jsonl');vv=np.array([r['conf'] or 0.0 for r in vals]);cc=np.array([int(r['label']==r['true']) for r in vals]);Pv=np.array([[r['dist'][cl] for cl in classes] for r in vals]);yv=np.array([ci[r['true']] for r in vals]);temp=float(rr['temperature'])
 Z=np.log(np.clip(Pv,1e-12,1))/temp;Z-=Z.max(1,keepdims=True);Pv=np.exp(Z);Pv/=Pv.sum(1,keepdims=True)
 sc=1-Pv[np.arange(len(vals)),yv]
 q=np.array([float(np.sort(sc[yv==i])[math.ceil((int((yv==i).sum())+1)*.95)-1]) for i in range(15)])
 with np.load(path,allow_pickle=False) as a:
  computed={key[2:]:metrics(a[key],a['correct']) for key in a.files if key.startswith('s_')}
  diffs={kind:{metric:abs(value-expected['methods'][kind][metric]) for metric,value in ms.items()} for kind,ms in computed.items()}
  localmax=max(d for vals2 in diffs.values() for metric,d in vals2.items() if metric!='ece_mass');maxdiff=max(maxdiff,localmax)
  massdiff=max(vals2['ece_mass'] for vals2 in diffs.values());maxmassdiff=max(maxmassdiff,massdiff)
  assert localmax<1e-12
  pred=lines(X/'out'/f'x1_{model}_{draw}_dec.jsonl');vt=np.array([r['conf'] or 0.0 for r in pred]);hp=np.clip(hist_predict(vt,hist_fit(vv,cc)),0,1)
  hist_error=float(np.max(np.abs(a['s_verbal_hist']-hp)));assert hist_error<1e-14
  sets=(1-a['Pt'])<=q[None,:];assert np.array_equal(sets,a['sets'])
  expected_set=expected['conformal']['lac_mondrian_0.05'];coverage=float(sets[np.arange(len(a['y'])),a['y']].mean());size=float(sets.sum(1).mean());assert abs(coverage-expected_set['coverage'])<1e-14 and abs(size-expected_set['mean_size'])<1e-14
  records[path.name]={'sha256':sha(path),'n':len(a['y']),'calibration_draw':valname,'calibration_n':len(vals),'metrics':computed,'reported_metric_absolute_differences':diffs,'max_absolute_reported_metric_difference_excluding_equal_mass_ECE':localmax,'max_equal_mass_ECE_difference':massdiff,'histogram_refit_prediction_max_absolute_difference':hist_error,'conformal_sets_exact_match':True,'coverage':coverage,'mean_set_size':size}
report={'status':'PASS_PRIMARY_POINT_METRICS_WITH_EQUAL_MASS_TIE_ORDER_LIMITATION','method':'Independent NumPy arithmetic on retained arrays/logs; no original script imports, no inference, no bootstrap repetitions. Tiny histogram calibration reconstructed directly.','numeric_tolerance':1e-12,'equal_mass_ECE_limitation':'Original code uses default unstabilized np.argsort, so score ties can be split differently between equal-count bins across NumPy versions. Diagnostic equal-mass ECE did not reproduce exactly; its differences are reported, never silently tolerated. The1e-12 gate is unchanged for fixed-bin ECE/AUROC/AURC and all other point metrics.','primary_validation_CV_means':cv_means,'CV_reported_mean_absolute_differences':cv_diffs,'primary_validation_CV_folds_provenance':'Audit derivative reconstructed from original StratifiedKFold seed0 specification; no originally saved fold-membership ledger. Agreement with three stored CV summaries independently supports reconstruction.','primary_winners_from_original_all_six_summary':{'calibrator':rep['primary_calibrator'],'ranking':rep['primary_ranking']},'stack_temperature_CV_optimizer_repeated':False,'stack_temperature_limitation':'Original sklearn/scipy fitting optimizers were not rerun. Their retained test scores are checked arithmetically; conformal quantiles are reconstructed with the retained temperature.','score_arrays':records,'global_max_absolute_point_metric_difference_excluding_equal_mass_ECE':maxdiff,'global_max_equal_mass_ECE_difference':maxmassdiff}
p=D/'metric_arithmetic_provenance_evidence_audit.json';p.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'report':str(p),'sha256':sha(p),'CV_means':cv_means,'global_max_error':maxdiff,'primary_test':records['x2_q8_s456_test_scores.npz'],'primary_realistic_n':records['x2_q8_s456_realistic_scores.npz']['n']},indent=2))
