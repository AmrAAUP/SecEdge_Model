"""Offline audit of recovered records; never imports or executes original code."""
from pathlib import Path
import json, hashlib, math, re, csv
from collections import Counter
import numpy as np

SRC=Path(r'F:\server_xai\FiveCapterThis\xai2026')
OUT=SRC/'out'
DEST=Path(__file__).parent
OLD=Path(r'C:\Users\MJO\Downloads\New folder (9)\revision_work\pi_evidence')
bindings={}
def bind(path):
    path=Path(path); b=path.read_bytes()
    bindings[str(path)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
    return b
def J(path): return json.loads(bind(path).decode('utf-8-sig'))
def JL(path): return [json.loads(s) for s in bind(path).decode('utf-8-sig').splitlines() if s.strip()]
def index(rows):
    out={r['row']:r for r in rows}; assert len(out)==len(rows), 'duplicate rows'
    return out
def close(a,b): return bool(np.allclose(a,b,atol=1e-12,rtol=1e-12))
def macro_f1(y,lab):
    vals=[]
    for c in range(15):
        tp=np.sum((y==c)&(lab==c)); denom=np.sum(y==c)+np.sum(lab==c)
        vals.append(2*tp/denom if denom else 0.)
    return float(np.mean(vals))
def auc(correct,score):
    positive=np.asarray(score)[np.asarray(correct,dtype=bool)]
    negative=np.asarray(score)[~np.asarray(correct,dtype=bool)]
    delta=positive[:,None]-negative[None,:]
    return float(np.mean((delta>0)+.5*(delta==0)))
def wilcox_approx(diff):
    # Independent signed-rank normal approximation, zero_method=wilcox,
    # average ranks and tie-adjusted variance, no continuity correction.
    diff=np.asarray(diff); diff=diff[diff!=0]; n=len(diff)
    if not n: return 1.
    ad=np.abs(diff); ranks=np.empty(n); tie_term=0
    for value in np.unique(ad):
        m=ad==value; count=int(m.sum()); ranks[m]=(np.sum(ad<value)+1+np.sum(ad<=value))/2
        tie_term+=count**3-count
    wp=float(ranks[diff>0].sum()); variance=(n*(n+1)*(2*n+1)-tie_term/2)/24
    z=(wp-n*(n+1)/4)/math.sqrt(variance)
    return math.erfc(abs(z)/math.sqrt(2))
def binom_p(a,b):
    n=a+b; k=min(a,b)
    return min(1.,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))
def ece(s,c):
    s=np.asarray(s); c=np.asarray(c)
    bins=np.clip(np.digitize(s,np.linspace(0,1,16)[1:-1],right=False),0,14)
    return float(sum(np.sum(bins==k)/len(c)*abs(c[bins==k].mean()-s[bins==k].mean()) for k in range(15) if np.any(bins==k)))
def aurc(s,c):
    err=1-np.asarray(c); order=np.argsort(-np.asarray(s),kind='stable')
    return float((np.cumsum(err[order])/np.arange(1,len(c)+1)).mean())
def temperature(P,T):
    Z=np.log(np.clip(P,1e-12,1))/T; Z-=Z.max(axis=1,keepdims=True)
    E=np.exp(Z); return E/E.sum(axis=1,keepdims=True)
def pi_distribution(rec,paths):
    norm={key:{int(k):v/sum(probs.values()) for k,v in probs.items()} for key,probs in rec['node_probs'].items()}
    result={}
    for label,path in paths.items():
        value=1.
        for i,token in enumerate(path):
            key=json.dumps(path[:i])
            if key in norm: value*=norm[key][token]
        result[label]=value
    mass=sum(result.values()); return {k:v/mass for k,v in result.items()}
def timing(rows):
    d={}
    for key in ['t_decide','t_trie','n_prompt']:
        vals=[r[key] for r in rows if key in r]
        if vals: d[key]={'n':len(vals),'mean':float(np.mean(vals)),'median':float(np.median(vals)),'min':float(min(vals)),'max':float(max(vals))}
    return d

B=J(OUT/'pi_bundle.json'); classes=list(B['paths']); CI={c:i for i,c in enumerate(classes)}
oldB=J(OLD/'xai_pi_bundle.json')
pi=index(JL(OLD/'xai_pi_results.jsonl')); newpi=index(JL(OUT/'pi_results.jsonl'))
gpu=index(JL(OUT/'x6_gpu_reference.jsonl'))
dec300=index(JL(OUT/'x1_q8_s456_test_sub300_dec.jsonl'))
decoded=index(J(OLD/'decoded_bundle_prompts.json')['flows'])
flows=index(B['flows']); rows=[f['row'] for f in B['flows']]
assert set(rows)==set(pi)==set(gpu)==set(newpi)==set(oldBf['row'] for oldBf in oldB['flows'])
assert len(rows)==150
input_checks=[]
for row in rows:
    f=flows[row]; g=gpu[row]; d=dec300[row]
    assert f['true']==g['true']==d['true']==decoded[row]['true']
    assert decoded[row]['prompt'].endswith(d['flow_block']), row
    assert f==next(f2 for f2 in oldB['flows'] if f2['row']==row)
    input_checks.append(row)
P=np.array([[pi_distribution(pi[r],B['paths'])[c] for c in classes] for r in rows])
G=np.array([[gpu[r]['dist_canonical'][c] for c in classes] for r in rows])
plabel=[json.loads(pi[r]['text'])['label'] for r in rows]
glabel=[gpu[r]['label'] for r in rows]; truth=[flows[r]['true'] for r in rows]
deltamax=np.abs(P.max(axis=1)-G.max(axis=1))
agree=np.array(plabel)==np.array(glabel)
ta=J(OUT/'x2_trackA_q8_s456_val_cal.json')
q=np.array(ta['tests']['test']['conformal']['lac_mondrian_0.05']['q'])
Pt=temperature(P,ta['temperature']); sets=1-Pt<=q[None,:]
covered=sets[np.arange(150),[CI[t] for t in truth]]; sizes=sets.sum(axis=1)
savedD=J(OUT/'x6_trackD.json')
parity={
 'n':150,'ten_per_true_class':dict(Counter(truth)),
 'all_150_unique_row_keys_and_truths_match':True,
 'all_150_decoded_retained_prompt_flow_blocks_match_gpu_input_blocks':True,
 'distinct_tokenized_inputs':len({tuple(flows[r]['ids']) for r in rows}),
 'retained_pi_results_byte_identical_to_recovered_original':bindings[str(OLD/'xai_pi_results.jsonl')]['sha256']==bindings[str(OUT/'pi_results.jsonl')]['sha256'],
 'retained_bundle_byte_identical_to_recovered_original':bindings[str(OLD/'xai_pi_bundle.json')]['sha256']==bindings[str(OUT/'pi_bundle.json')]['sha256'],
 'agreement_count':int(agree.sum()),'label_agreement':float(agree.mean()),
 'mean_abs_delta_pmax':float(deltamax.mean()),'max_abs_delta_pmax':float(deltamax.max()),
 'mean_absolute_difference_over_all_15_class_probabilities':float(np.abs(P-G).mean()),
 'pi_accuracy_k':sum(a==t for a,t in zip(plabel,truth)),'gpu_accuracy_k':sum(a==t for a,t in zip(glabel,truth)),
 'mismatch_rows':[{'row':r,'true':truth[i],'pi_label':plabel[i],'gpu_label':glabel[i]} for i,r in enumerate(rows) if not agree[i]],
 'pi_conformal_covered_k':int(covered.sum()),'pi_conformal_coverage':float(covered.mean()),
 'pi_mean_set_size':float(sizes.mean()),'pi_singletons_k':int((sizes==1).sum()),'pi_empty_sets_k':int((sizes==0).sum()),
 'temperature':ta['temperature'],'mondrian_thresholds':q.tolist(),
 'pi_component_timings_s':timing(list(pi.values())),
 'trie_over_decision_mean_ratio':float(np.mean([pi[r]['t_trie'] for r in rows])/np.mean([pi[r]['t_decide'] for r in rows])),
 'all_saved_trackD_metrics_recomputed_match':all(close(v,savedD[k]) for k,v in {
 'label_agreement_gpu_pi':agree.mean(),'mean_abs_delta_pmax':deltamax.mean(),'max_abs_delta_pmax':deltamax.max(),
 'pi_accuracy':np.mean(np.array(plabel)==np.array(truth)), 'gpu_accuracy':np.mean(np.array(glabel)==np.array(truth)),
 'pi_conformal_coverage_0.05':covered.mean(),'pi_mean_set_size':sizes.mean(),'pi_singleton_rate':(sizes==1).mean(),
 'pi_decide_s_mean':np.mean([pi[r]['t_decide'] for r in rows]),'pi_trie_s_mean':np.mean([pi[r]['t_trie'] for r in rows])}.items()),
 'probability_delta_definition':'Mean absolute difference between each device\'s maximum canonical grammar-normalized class probability, not mean absolute error over the 15-vector.',
 'scope':'Same seed456 Q8_0 checkpoint across GPU-configured workstation and Pi; not BF16-versus-Q8 and not per-class empirical95% guarantee.'}

models={}; arrays={}; records={}
for model,draw,val in [('q8_s456','test_sub300','val_sub300'),('bf16_s456','test_sub300','val_sub300'),('q8_s456','test_sub1500','val_sub1500'),('q4_s456','test_sub1500','val_sub1500')]:
    key=model+'_'+draw
    rs=JL(OUT/f'x1_{model}_{draw}_dec.jsonl'); recs=index(rs); records[key]=recs
    path=OUT/f'x2_{model}_{draw}_scores.npz'; bind(path)
    with np.load(path,allow_pickle=False) as z: z={k:z[k].copy() for k in z.files}
    arrays[key]=z
    s=J(OUT/f'x2_trackA_{model}_{val}.json'); summary=s['tests'][draw]
    y=np.array([CI[r['true']] for r in rs]); lab=np.array([CI.get(r['label'],-1) for r in rs]); correct=lab==y
    assert np.array_equal(y,z['y']) and np.array_equal(lab,z['lab']) and np.array_equal(correct,z['correct'])
    Praw=np.array([[r['dist'][c] for c in classes] for r in rs]); Ps=temperature(Praw,s['temperature'])
    threshold=np.array(summary['conformal']['lac_mondrian_0.05']['q'])
    assert close(Ps,z['Pt']) and np.array_equal((1-Ps)<=threshold,z['sets'])
    sizes=z['sets'].sum(axis=1); coverage=z['sets'][np.arange(len(y)),y]
    pc=s['primary_calibrator']; pr=s['primary_ranking']
    pcscore=z['s_'+pc]; prscore=z['s_'+pr]
    models[key]={
      'n':len(rs),'unique_rows':len(recs),'npz_row_order_truth_label_correct_matches_decision_records':True,
      'npz_Pt_and_primary_sets_match_recomputed_temperature_and_frozen_thresholds':True,
      'accuracy':float(correct.mean()),'correct_k':int(correct.sum()),
      'macro_f1':macro_f1(y,lab),
      'raw_verbal_ece':ece(z['s_verbal_raw'],correct),'primary_calibrator':pc,'primary_ece':ece(pcscore,correct),
      'primary_ranking':pr,'primary_auroc':auc(correct,prscore),'primary_aurc':aurc(prscore,correct),
      'coverage':float(coverage.mean()),'coverage_k':int(coverage.sum()),
      'mean_set_size':float(sizes.mean()),'singleton_rate':float((sizes==1).mean()),
      'empty_rate':float((sizes==0).mean()),'temperature':s['temperature'],
      'saved_trackA_primary_metrics_match':all(close(v,w) for v,w in [(correct.mean(),summary['accuracy']),
        (ece(pcscore,correct),summary['methods'][pc]['ece']),
        (auc(correct,prscore),summary['methods'][pr]['auroc']),
        (aurc(prscore,correct),summary['methods'][pr]['aurc']),
        (coverage.mean(),summary['conformal']['lac_mondrian_0.05']['coverage']),
        (sizes.mean(),summary['conformal']['lac_mondrian_0.05']['mean_size'])]),
      'component_timing':timing(rs)}

pairing={}
for first,second in [('q8_s456_test_sub300','bf16_s456_test_sub300'),('q8_s456_test_sub1500','q4_s456_test_sub1500')]:
    a,b=records[first],records[second]; assert set(a)==set(b)
    ordered=list(a); assert ordered==list(b)
    assert all(a[r]['true']==b[r]['true'] and a[r]['flow_block']==b[r]['flow_block'] for r in ordered)
    sa=arrays[first]['sets'].sum(axis=1); sb=arrays[second]['sets'].sum(axis=1)
    diff=sb-sa
    pairing[first+'_vs_'+second]={
      'n':len(ordered),'row_order_identical':True,'same_exact_input_block_and_truth_for_every_pair':True,
      'label_agreement_k':sum(a[r]['label']==b[r]['label'] for r in ordered),
      'label_disagreements':[r for r in ordered if a[r]['label']!=b[r]['label']],
      'mean_set_size_second_minus_first':float(diff.mean()),
      'sets_identical_k':int((arrays[first]['sets']==arrays[second]['sets']).all(axis=1).sum()),
      'wilcoxon_set_size_p_independent_tie_adjusted_normal_approximation':wilcox_approx(diff)}

validation_pairing={}
for first,second,draw in [('q8_s456','bf16_s456','val_sub300'),('q8_s456','q4_s456','val_sub1500')]:
    a=index(JL(OUT/f'x1_{first}_{draw}_dec.jsonl')); b=index(JL(OUT/f'x1_{second}_{draw}_dec.jsonl'))
    assert list(a)==list(b)
    assert all(a[r]['true']==b[r]['true'] and a[r]['flow_block']==b[r]['flow_block'] for r in a)
    validation_pairing[first+'_vs_'+second]={'draw':draw,'n':len(a),'same_rows_order_input_blocks_and_truth':True,
      'calibration_scope':'Each checkpoint independently selects/refits calibration on this shared validation cohort. Comparison does not hold the fitted calibrator fixed across precision.'}

audit={}
for model in ['q8_s456','q4_s456']:
    rs=JL(OUT/f'x3_audit_{model}_rows.jsonl'); n=sum(r['parsed'] for r in rs)
    accepted=[r for r in rs if r['parsed']]; audit[model]={
      'total_rows':len(rs),'parsed':n,'unparsed':len(rs)-n,
      'automatic_literal_hallucination_k':sum(r['has_hallucination'] for r in accepted),
      'automatic_unobservable_k':sum(r['has_unobservable'] for r in accepted),
      'automatic_literal_hallucination_rate':sum(r['has_hallucination'] for r in accepted)/n,
      'automatic_unobservable_rate':sum(r['has_unobservable'] for r in accepted)/n}
    audit[model]['byrow']=index(rs)
aq=audit['q8_s456'].pop('byrow'); ab=audit['q4_s456'].pop('byrow')
common=[r for r in aq if aq[r]['parsed'] and ab[r]['parsed']]; assert len(common)==1485
only4=sum(ab[r]['has_unobservable'] and not aq[r]['has_unobservable'] for r in common)
only8=sum(aq[r]['has_unobservable'] and not ab[r]['has_unobservable'] for r in common)
audit['paired_automatic_unobservable']={'n_both_parsed':len(common),'only_q4':only4,'only_q8':only8,
    'mcnemar_exact_p':binom_p(only4,only8),
    'scope':'Stored automatic regex/field-check flags; unsupported semantics require separate documented manual audit.'}
audit['saved_exploratory_integrity']=J(OUT/'x3_exploratory_integrity.json')
audit['scope']='Literal value checks, automatic unobservable-category screening and exploratory technique-catalogue/label checks are distinct outcomes. Do not equate the lower automatic unsupported rate for Q4 with better semantic faithfulness.'

full_timing={}
for model in ['q8_s456','q4_s456']:
    rs=JL(OUT/f'x1_{model}_test_sub1500_full.jsonl')
    dec=records[model+'_test_sub1500']; assert set(dec)==set(index(rs))
    assert all(dec[r['row']]['flow_block']==r['flow_block'] and dec[r['row']]['true']==r['true'] for r in rs)
    full_timing[model]={'n':len(rs),'same_inputs_as_decision_cohort':True,'full_json_generation_wall_seconds':timing(rs),
      'max_new_tokens_from_source':384,'includes_full_rationale_action_action_rationale':True,'location':'workstation configured for CUDA offload, not Pi'}

source_notes={}
for f in ['x6_pi.py','x4_counterfactual.py','xai_common.py','x1_generate.py','x2_trackA.py','x5_compare.py','model_hashes.json','x0_gates.json','x4_trackB.json','x4_run.log']:
    path=(OUT/f) if f.endswith(('.json','.log')) else SRC/f
    source_notes[f]=bind(path).decode('utf-8-sig',errors='replace')
logs={}
for model in ['q8_s456','bf16_s456','q4_s456']:
    s=bind(OUT/f'server_{model}.log').decode('utf-8',errors='replace')
    ls=s.splitlines()
    logs[model]={
     'model_load_lines':[l for l in ls if 'load_model: loading model' in l],
     'context_lines':[l for l in ls if 'n_ctx_slot' in l][:3],
     'gpu_hardware_or_build_lines':[l for l in ls if any(p in l.lower() for p in ['nvidia','cuda','offload','n_gpu','build:','system_info:'])][:10],
     'limitation':'This saved log identifies the loaded model path/context but contains no GPU name, backend/offload confirmation, llama.cpp build commit or GPU timing build metadata.'}

report={
 'audit_type':'Independent read-only offline GPU/Pi pairing, quantization and cost audit of newly supplied original records',
 'created_date':'2026-10-08','source_root':str(SRC),
 'execution_constraints':'Original modules were never imported or executed; no model runs, source mutations, pickle reads, network operations or calibration refits. NPZ decoded with allow_pickle=False. Only recovered record arithmetic and literal source inspection.',
 'gpu_pi_parity':parity,'quantization_models':models,'quantization_pairing':pairing,'quantization_validation_pairing':validation_pairing,'rationale_audit':audit,
 'full_generation_timing':full_timing,'saved_server_log_metadata':logs,
 'producer_bindings':{
  'gpu_reference_producer':'x4_counterfactual.py gpuref(), lines137–148; called in run() after verify_model(q8_s456) and Server(q8_s456), lines162–167. Input from original pi_bundle row IDs and original test frame; canonical-prefix distribution saved with label/conf.',
  'gpu_server_configuration':'xai_common.py lines39,148–155: CUDA llama-server.exe; default -ngl99, ctx2048, one slot, four CPU threads. x1_generate.py132,150 overrides BF16 to -ngl20 (partial offload); Q8/Q4 request99. This config alone is not a measured GPU model/offload/build certificate.',
  'pi_worker':'out/xai_pi_run.py and embedded BOARD_RUNNER in x6_pi.py; Llama n_ctx2048/n_threads4/n_batch512, decision max_tokens48, temp0, reset before each flow and separately timed trie evaluation.',
  'expected_and_cached_model_digests':J(OUT/'model_hashes.json'),
  'model_digest_scope':'Expected hashes and cached hashes are retained documentation; this audit did not read multi-GB models. The Q8 seed456 digest c7aaaeefa586f616a1d91fbd007cfbfc7e1844baf7a399e6809cf4d24745d40a agrees with the earlier independently verified actual Pi model digest.'},
 'cost_limits':{
  'measured_original_pi_components':'150 short grammar decisions and separate class-distribution scoring; excludes attribution, full rationale, model loading, tokenization/rendering and cascade/retrieval overhead.',
  'evidence_text_formatting_saved_mean_ms':J(OUT/'x4_trackB.json')['B-T3']['text_generation_ms_mean'],
  'gate_shap_saved_mean_ms_workstation':J(OUT/'x4_trackB.json')['B-T4']['shap_ms_per_flow_workstation'],
  'counterfactual_run_log_terminal':'[10:04:19] counterfactual750/750 2.3s/flow calls=10947; GPU-configured workstation, cumulative rounded wall/cached scorer evaluations, not Pi inference or exact per-record timings.',
  'cannot_add_component_costs':'Formatting0.002923ms excludes attribution; separate SHAP timing is workstation-based and not an adequate proxy for SLM field sensitivity. GPU attribution ~2.3s/flow and shortPi scoring are different workflows/locations; no complete original Pi attribution+rationale cost or physical-call ledger recovered here.',
  'pi_full_rationale_28_to_42_seconds':'Not established by these recovered original Pi150 component records. Full-generation saved rows are workstationQ8/Q4 with384-token cap. Earlier thesis timing claims should remain separately sourced/scoped.',
  'bf16_speed_comparison_not_precision_only':'BF16 requests20GPU layers while Q8/Q4 request99. Different offload configurations mean the saved BF16-versus-Q8 wall-time difference is not a controlled precision-only speedup.',
  'chronology':'Retained result timestamps, filesystem times and script identities support provenance mapping; they do not independently authenticate historical preregistration chronology.'},
 'recommended_claims':[
  'On the same150 keyed inputs (138 distinct tokenized inputs;10perclass), the retained seed456 Q8_0 GPU/Pi records agree on149labels (99.33%); mean absolute change in maximum canonical class probability is0.004526.',
  'Distribution scoring adds1.149s to27.560s mean short-decision latency on the retainedPi sample (4.17% of decision time), excluding full rationale and attribution.',
  'BF16 andQ8 have equal92% accuracy and identical predicted labels on the paired300-flow sample, while their selected calibration and set metrics differ slightly.',
  'On the paired1500-flow sample Q4 enlarges mean prediction sets from1.223 to1.665 (+0.443; savedbootstrapCI0.408–0.477), while accuracy falls from91.13% to86.13%; its automatic unsupported screen rate is lower but exploratory catalogue/label consistency checks degrade.',
  'The frozenGPUcalibrator applied toPi probabilities covers133/150truths (88.67%); these records do not establish95% on-device empirical coverage.'],
 'source_bindings':bindings}

# Verify every read source stayed byte-identical before writing a derived report.
for path,b in bindings.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==b['sha256'],path
report['all_read_source_hashes_unchanged_after_audit']=True
dest=DEST/'gpu_precision_cost_audit_locate_original_20261008.json'
dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
readable='''Original XAI GPU, precision and component-cost verification
Read-only audit: 8 October 2026

Recovered source: F:\\server_xai\\FiveCapterThis\\xai2026
No original code was executed/imported and no model was run. Original records
remained unchanged. Arithmetic was implemented independently; NPZ arrays were
read without executable pickle support. Full record hashes are in the JSON audit.

GPU–Pi comparison
The recovered original Pi bundle and result file are byte-identical to the
previously retained copies. All 150 unique row keys and true labels match the
GPU reference and Q8 input records. Each decoded retained prompt matches its
GPU input block. There are 138 distinct tokenized inputs and ten records per
class. The GPU producer is x4_counterfactual.py, gpuref(), lines 137–148;
run() verifies seed456 Q8_0 and starts the server before producing the reference.

Independent recomputation gives 149/150 label agreement (99.333%), mean absolute
change in maximum canonical class probability 0.00452590873863642 and maximum
change 0.0584416261011943. This is the difference between two maximum class
probabilities, not average error over all fifteen probabilities. The only label
disagreement is row 730 (truth/Pi: DDoS_HTTP; GPU: Password). Pi accuracy is
135/150 and GPU accuracy 134/150. Every saved Track D metric is reproduced.

The GPU-fitted temperature and classwise thresholds transferred to Pi cover
133/150 truths (88.67%); sets average 1.2533 labels, with 107 singletons and
seven empty sets. Label agreement therefore does not prove 95% coverage transfer.

Quantization comparisons
BF16 and Q8 use exactly the same 300 test inputs and the same 300 validation
inputs. All 300 predicted labels agree; accuracy is 276/300 (92.0%) and macro-F1
0.918763 for both. Each precision independently selects/refits calibration.
Q8 selects stack calibration (ECE 0.032601), BF16 verbal histogram calibration
(ECE 0.022650). Set means are 1.25 versus 1.24 and coverage 289/300 versus287/300.
These are close descriptive results, rather than identical calibrated outcomes.

Q8 and Q4 share all 1,500 test inputs and 1,500 validation inputs. All NPZ truth,
label, correctness, temperature-scaled probability and prediction-set arrays
match the keyed records and frozen thresholds. Q8 accuracy is 91.133%, Q4
86.133%; 1,375/1,500 predicted labels agree. Mean set size increases from1.222667
to1.665333 (paired difference0.442667). Independent tie-adjusted signed-rank
normal calculation reproduces p=2.87199822422598e-99. The saved bootstrap
interval is[0.408,0.477333]; this audit did not repeat that bootstrap.

The automatic audit parses 1,499 Q8 and1,486 Q4 rationales. Literal-fabrication
flags are0/1,499 versus9/1,486; automatic unobservable flags945/1,499(63.04%)
versus847/1,486(57.00%). On1,485 jointly parsed records, only-Q4 flags122 and
only-Q8 flags208 reproduce McNemar p=2.54617870131771e-6. Thus the composite
target requiring increased automatic flags remains unmet. Saved exploratory
catalogue/label consistency checks separately report317/1,486 Q4 contradictions
(21.33%) and51/1,486 nonexistent technique identifiers(3.43%), versus zero for
Q8. These categories must not be conflated with manual unsupported-claim coding.

Costs and runtime evidence
Pi short decisions average27.560347s and separately timed class scoring1.149080s;
their ratio is4.1693%. The decision cap is48 tokens. These measurements exclude
attribution, full rationale generation, loading, tokenization/rendering and
cascade/retrieval overhead. Recovered full-JSON records are workstation runs,
with384-token cap: Q8 mean2.683543s andQ4 mean2.170667s over1,500 inputs.

Source requests CUDA llama-server, context2048, four threads and GPU layers99
forQ8/Q4. BF16 requests only20 layers, so its3.492567s decision mean versus
Q8's0.661470s on300 inputs cannot establish a precision-only speedup. Logs
confirm loaded checkpoint paths and2048 context; they omit GPU name, actual
offload/backend confirmation, build commit and GPU-hardware configuration.

Evidence-text formatting is a saved0.002923ms mean AFTER attribution. TreeSHAP
42.540516ms is a workstation wall mean over50 repetitions of one flow, not a
Pi measurement or GPU-kernel timing. The original GPU-host counterfactual log
ends at750 flows,10,947 scorer evaluations and rounded2.3s/flow; caching and
cumulative timing prevent treating this as a physical-call or per-flow ledger.
No original complete Pi attribution+rationale timing is recovered here. The
claimed28–42s full Pi rationale latency is not proved by this short-decision
file. Original chronology requires separate registration evidence evaluation.
'''
txt=DEST/'gpu_precision_cost_audit_locate_original_20261008.txt'
txt.write_text(readable,encoding='utf-8')
print(json.dumps({'report':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
 'readable_report':str(txt),'readable_sha256':hashlib.sha256(txt.read_bytes()).hexdigest(),
 'parity':parity,'models':models,'pairs':pairing,'rationale':audit,'timing':full_timing},ensure_ascii=False,indent=2))
