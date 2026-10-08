"""Analyze completed strict-reset v2/v3 sensitivity chain; scientific v2 plan unchanged."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse, csv, hashlib, itertools, json, math
import numpy as np

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path(r'C:\Users\MJO\Downloads\New folder (8)\review_revision')
EXPECTED_PROTOCOL_SHA = '47a421ec04a238ab7ac06d4c006ffa525e4ac2008e610066ac11e9a6b05b727d'
EXPECTED_AMENDMENT_SHA = 'bf22398860adc17055712454be6009c89279f4decdc5ee95e96e2f3757046fa1'
FLOOR = 1e-12
BOOT_SEED = 20261008
BOOT_REPS = 10000

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def text_sha(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump_exclusive(name, data):
    path = ROOT/name
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    return path

def parse(block):
    return dict(x.split('=', 1) for x in block.split('\n', 1)[1].split('  '))
def render(values, order):
    return 'Flow features:\n'+'  '.join(k+'='+values[k] for k in order if k in values)
def intervene(values, donor, selection, order):
    changed = dict(values)
    for key in selection:
        if key in donor: changed[key] = donor[key]
        else: changed.pop(key, None)
    return render(changed, order)
def validity(values):
    issues = []
    for key in ('tcp_srcport', 'tcp_dstport'):
        if key in values:
            try:
                value=float(values[key])
                if not math.isfinite(value) or not value.is_integer() or not 0<=value<=65535:
                    issues.append(key+':invalid_port')
            except (ValueError, OverflowError): issues.append(key+':non_numeric_port')
    try:
        flags=float(values.get('tcp_flags', '0')); ack=float(values.get('tcp_flags_ack', '0'))
        if not math.isfinite(flags) or not flags.is_integer() or not 0<=flags<=511:
            issues.append('tcp_flags_invalid')
        elif not math.isfinite(ack) or ack not in (0,1): issues.append('tcp_ack_invalid')
        elif bool(int(flags)&16)!=bool(int(ack)):
            issues.append('tcp_ack_flag_disagreement_under_zero_omission')
    except (ValueError, OverflowError): issues.append('tcp_flag_not_numeric')
    return issues

def tied_ranks(values):
    values=np.asarray(values,dtype=float); indices=np.argsort(values,kind='stable')
    ranks=np.empty(len(values),dtype=float); start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[indices[end]]==values[indices[start]]: end+=1
        ranks[indices[start:end]]=(start+1+end)/2
        start=end
    return ranks

def spearman_tied(a,b):
    ar=tied_ranks(a); br=tied_ranks(b); ar-=ar.mean();br-=br.mean()
    den=math.sqrt(float(ar@ar)*float(br@br))
    if den==0:return None
    return float(np.clip((ar@br)/den,-1,1))

def cluster_summary(flow_rows, metric):
    rows=[r for r in flow_rows if r.get(metric) is not None and math.isfinite(r[metric])]
    if not rows:return {'n_flows':0,'n_clusters':0,'mean':None,'mean_ci95':None}
    groups=defaultdict(list)
    for row in rows:groups[row['input_sha256']].append(float(row[metric]))
    keys=sorted(groups); sums=np.array([sum(groups[k]) for k in keys]); counts=np.array([len(groups[k]) for k in keys])
    rng=np.random.default_rng(BOOT_SEED)
    draws=rng.integers(0,len(keys),size=(BOOT_REPS,len(keys)))
    boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    vals=np.array([r[metric] for r in rows],dtype=float)
    return {'n_flows':len(rows),'n_clusters':len(keys),'mean':float(vals.mean()),'median':float(np.median(vals)),
            'min':float(vals.min()),'max':float(vals.max()),'mean_ci95':[float(x) for x in np.quantile(boot,[.025,.975])],
            'bootstrap_replicates':BOOT_REPS,'bootstrap_seed':BOOT_SEED,
            'method':'Percentile cluster bootstrap by exact original input hash; sample contributing clusters with replacement and retain every occurrence within each sampled cluster.'}

def inventory(rows):
    totals=Counter(r['event'] for r in rows)
    return {'n_events':len(rows),'event_counts':dict(totals)}


def event_key(row):
    kind=row['event']
    if kind in ('baseline','flow_complete'):values=[kind,row['flow_id']]
    elif kind=='single':values=[kind,row['flow_id'],row['donor_index'],row['field']]
    elif kind=='joint':values=[kind,row['flow_id'],row['donor_index'],row['method']]
    else:return None
    return json.dumps(values,separators=(',',':'))

def verify_chain_cache(prefix, continuation):
    score_events={'baseline','single','joint'}
    old_seen=set();historical_calls=0;old_metric_keys=set()
    prior_baselines={x['flow_id']:x for x in prefix if x['event']=='baseline'}
    for row in prefix:
        key=event_key(row)
        if key is not None:
            assert key not in old_metric_keys;old_metric_keys.add(key)
        if row['event'] in score_events:
            h=row['input_sha256'];assert row['executed']==(h not in old_seen)
            old_seen.add(h);historical_calls+=int(row['executed'])
    full={};focal={}
    # Match the continuation worker's exact two-pass prior-cache hydration.
    for row in prefix:
        if row['event']=='baseline':
            h=row['input_sha256'];key=event_key(row)
            if h in full:assert full[h][0]==row['q']
            else:full[h]=(row['q'],key,'v2')
            c=row['focus_class'];focal[h,c]=(row['q'][c],key,'v2')
    for row in prefix:
        if row['event'] in ('single','joint'):
            h=row['input_sha256'];c=prior_baselines[row['flow_id']]['focus_class'];key=event_key(row)
            if (h,c) in focal:assert focal[h,c][0]==row['q_focus']
            else:focal[h,c]=(row['q_focus'],key,'v2')
            if h in full:assert full[h][0][c]==row['q_focus']
    hydrated_full=len(full);hydrated_focal=len(focal)
    evaluated_hashes=set(old_seen);continuation_calls=0;focus_by_flow={k:v['focus_class'] for k,v in prior_baselines.items()}
    all_keys=set(old_metric_keys);cache_sources=Counter();new_score_keys=set();reevaluated_old_hashes=set()
    for row in continuation:
        assert row['execution_phase']=='v3' and row['execution_amendment_sha256']==EXPECTED_AMENDMENT_SHA
        key=event_key(row)
        if key is not None:
            assert key not in all_keys,'A prior or continuation metric event was repeated'
            assert row['event_key']==key
            all_keys.add(key)
        if row['event']=='execution_resume':
            assert row['historical_model_evaluations']==historical_calls
            assert row['historical_distinct_evaluated_hashes']==len(old_seen)
            assert row['prior_full_q_cache_entries']==hydrated_full and row['prior_focal_cache_entries']==hydrated_focal
            assert row['prior_event_keys']==len(old_metric_keys)
            assert row['completed_flow_ids']==[1]
        if row['event'] in score_events:
            h=row['input_sha256'];kind=row['event'];origin=row['cache_source_event_key'];source=row['cache_source']
            c=None if kind=='baseline' else focus_by_flow[row['flow_id']]
            fresh=row['executed'];assert isinstance(fresh,bool);cache_sources[source]+=1
            if fresh:
                assert source=='full_reset_evaluation' and origin==key
                assert h not in full and (c is None or (h,c) not in focal)
                q=row['q'] if kind=='baseline' else row['q_vector']
                assert len(q)==15 and abs(sum(q.values())-1)<1e-10
                assert all(math.isfinite(v) and 0<=v<=1 for v in q.values())
                for (old_h,old_c),(old_v,old_origin,old_phase) in focal.items():
                    if old_h==h:assert q[old_c]==old_v,'Recomputed vector changed a retained focal value'
                if h in old_seen:reevaluated_old_hashes.add(h)
                continuation_calls+=1;evaluated_hashes.add(h)
                full[h]=(q,key,'v3')
                for cls,value in q.items():focal[h,cls]=(value,key,'v3')
                expected=q if kind=='baseline' else q[c]
            else:
                assert 'q_vector' not in row
                if h in full:
                    q,expected_origin,phase=full[h]
                    expected_source='prior_baseline_full' if phase=='v2' else 'continuation_full'
                    assert source==expected_source and origin==expected_origin
                    expected=q if kind=='baseline' else q[c]
                else:
                    assert c is not None and (h,c) in focal
                    expected,expected_origin,phase=focal[h,c]
                    expected_source='prior_focal_scalar' if phase=='v2' else 'continuation_focal_scalar'
                    assert source==expected_source and origin==expected_origin
                assert origin in all_keys and origin!=key
            if kind=='baseline':
                assert row['q']==expected
                focus_by_flow[row['flow_id']]=row['focus_class']
            else:assert row['q_focus']==expected
            new_score_keys.add(key)
        if row['event'] in ('flow_complete','complete','execution_progress'):
            assert row['historical_model_evaluations']==historical_calls
            assert row['continuation_model_evaluations']==continuation_calls
            assert row['actual_model_evaluations_total']==historical_calls+continuation_calls
            assert row['distinct_input_hashes_evaluated_total']==len(evaluated_hashes)
            assert row['elapsed_scope']=='continuation'
            if 'distinct_model_evaluations' in row:assert row['distinct_model_evaluations']==len(evaluated_hashes)
    return {'historical_model_evaluations':historical_calls,'continuation_model_evaluations':continuation_calls,
            'actual_model_evaluations_total':historical_calls+continuation_calls,
            'distinct_input_hashes_evaluated_total':len(evaluated_hashes),
            'old_historical_hashes_recomputed_for_full_vector':len(reevaluated_old_hashes),
            'prior_metric_keys':len(old_metric_keys),'new_score_event_keys':len(new_score_keys),
            'continuation_cache_sources':dict(cache_sources),
            'counter_scope':'Recorded score evaluations across the retained prefix and continuation. Unlogged interrupted work and the rejected optimization benchmark are excluded.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exit-verification',required=True,type=Path,
        help='Completed pi_sensitivity_terminal_v3.json: exit0, immutable prefix identity, complete marker and hashes verified.')
    args=parser.parse_args()
    assert ROOT==EXPECTED_ROOT,'Outputs must remain exclusively in authorized New folder (8)'
    assert sha(ROOT/'INTERVENTION_PROTOCOL_v2.json')==EXPECTED_PROTOCOL_SHA
    protocol=load(ROOT/'INTERVENTION_PROTOCOL_v2.json'); payload=load(ROOT/'pi_sensitivity_payload_v2.json')
    assert payload['protocol_sha256']==EXPECTED_PROTOCOL_SHA
    assert sha(ROOT/'pi_sensitivity_remote_v2.py')==protocol['runner_sha256']
    assert sha(ROOT/'benign_validation_donors.json')==protocol['donor_manifest_sha256']
    amendment=load(ROOT/'EXECUTION_CONTINUATION_v3.json')
    assert sha(ROOT/'EXECUTION_CONTINUATION_v3.json')==EXPECTED_AMENDMENT_SHA
    assert amendment['scientific_protocol_sha256']==EXPECTED_PROTOCOL_SHA
    assert sha(ROOT/'pi_sensitivity_remote_v3.py')==amendment['v3_worker_sha256']
    assert sha(ROOT/'pi_sensitivity_payload_v2.json')==amendment['v2_payload_sha256']
    ledger=load(ROOT/'continuation_input_event_ledger_v3.json')
    assert sha(ROOT/'continuation_input_event_ledger_v3.json')==amendment['input_event_ledger_sha256']
    payload3=load(ROOT/'pi_sensitivity_payload_v3.json')
    assert payload3['execution_amendment_sha256']==EXPECTED_AMENDMENT_SHA
    assert {k:v for k,v in payload3.items() if k not in ('execution_amendment_sha256','immutable_v2_prefix_sha256')}==payload
    exit_evidence=load(args.exit_verification)
    assert exit_evidence['remote_exit_code']==0,'Successful launcher exit must be observed before analysis'
    assert exit_evidence['scientific_protocol_sha256']==EXPECTED_PROTOCOL_SHA
    assert exit_evidence['execution_amendment_sha256']==EXPECTED_AMENDMENT_SHA
    assert exit_evidence['has_final_complete'] and exit_evidence['last_event']=='complete'
    assert exit_evidence['immutable_v2_raw_unchanged'] and exit_evidence['merged_prefix_byte_identical']
    assert exit_evidence['worker_sha256']==amendment['v3_worker_sha256']
    result_path=ROOT/'pi_sensitivity_results_v3.jsonl'
    raw_bytes=result_path.read_bytes();raw=raw_bytes.decode('utf-8')
    assert sha(result_path)==exit_evidence['merged_results_sha256']
    assert len(raw_bytes)==exit_evidence['merged_results_bytes']
    old_path=ROOT/'pi_sensitivity_results_v2.jsonl';prefix_bytes=old_path.read_bytes()
    assert sha(old_path)==amendment['immutable_v2_prefix_sha256']==exit_evidence['immutable_v2_raw_sha256']
    assert len(prefix_bytes)==amendment['immutable_v2_prefix_bytes']==exit_evidence['old_prefix_bytes']
    assert raw_bytes[:len(prefix_bytes)]==prefix_bytes
    prefix=[json.loads(x) for x in prefix_bytes.splitlines()];assert len(prefix)==86
    continuation=[json.loads(x) for x in raw_bytes[len(prefix_bytes):].splitlines()]
    assert len(continuation)==exit_evidence['new_events']
    assert [event_key(x) for x in prefix if event_key(x) is not None]==ledger['prior_event_keys']
    assert raw.endswith('\n'),'Run output must have a complete final JSONL record'
    lines=[json.loads(x) for x in raw.splitlines() if x.strip()]
    assert lines[-1]['event']=='complete','Never analyze partial results as final'
    complete=[r for r in lines if r['event']=='complete'];assert len(complete)==1
    start=[r for r in lines if r['event']=='starting'];assert len(start)==1
    assert start[0]['protocol_sha256']==EXPECTED_PROTOCOL_SHA
    assert start[0]['boot_id']==protocol['hardware']['boot_id']
    phase_start=[x for x in continuation if x['event']=='execution_start'];assert len(phase_start)==1
    assert phase_start[0]['protocol_sha256']==EXPECTED_PROTOCOL_SHA
    assert phase_start[0]['immutable_v2_prefix_sha256']==amendment['immutable_v2_prefix_sha256']
    assert phase_start[0]['immutable_v2_prefix_events']==86
    assert phase_start[0]['boot_id']==protocol['hardware']['boot_id']
    assert exit_evidence['handle']['remote']==phase_start[0]
    assert complete[0]==exit_evidence['completion']
    assert complete[0]['protocol_sha256']==EXPECTED_PROTOCOL_SHA and complete[0]['prefix_reuse'] is False
    assert complete[0]['immutable_v2_prefix_sha256']==amendment['immutable_v2_prefix_sha256']
    phase_tokens=[x for x in continuation if x['event']=='execution_tokenization_verified'];assert len(phase_tokens)==1
    assert phase_tokens[0]['retained_prompts']==phase_tokens[0]['exact_id_matches']==150
    assert phase_tokens[0]['bos_added'] is False
    phase_models=[x for x in continuation if x['event']=='execution_model_loaded'];assert len(phase_models)==1
    assert phase_models[0]['model_sha256']==protocol['model_sha256']
    assert phase_models[0]['target_flows']==30 and phase_models[0]['donors']==5
    assert phase_models[0]['n_ctx']==2048 and phase_models[0]['n_threads']==4 and phase_models[0]['n_batch']==512
    assert phase_models[0]['logits_all'] is True and phase_models[0]['prefix_reuse'] is False
    reused=[x for x in continuation if x['event']=='execution_flow_reused']
    assert len(reused)==1 and reused[0]['flow_id']==1
    assert reused[0]['prior_flow_complete_event_key']==event_key({'event':'flow_complete','flow_id':1})
    cache_audit=verify_chain_cache(prefix,continuation)

    verified=[r for r in lines if r['event']=='tokenization_verified'];assert len(verified)==1
    assert verified[0]['exact_id_matches']==verified[0]['retained_prompts']==150
    assert verified[0]['bos_added'] is False
    model_loaded=[r for r in lines if r['event']=='model_loaded'];assert len(model_loaded)==1
    assert model_loaded[0]['model_sha256']==protocol['model_sha256']
    assert model_loaded[0]['target_flows']==30 and model_loaded[0]['donors']==5
    targets={f['i']:f for f in payload['flows']};assert len(targets)==30
    target_manifest={f['i']:f for f in protocol['target_indices']};assert set(targets)==set(target_manifest)
    for fid,flow in targets.items():
        assert text_sha(flow['flow_block'])==target_manifest[fid]['input_sha256']
        assert flow['source_row']==target_manifest[fid]['source_row'] and flow['true']==target_manifest[fid]['true']
    clusters=Counter(text_sha(f['flow_block']) for f in targets.values());assert len(clusters)==28
    assert sorted(Counter(f['true'] for f in targets.values()).values())==[2]*15
    order=payload['field_order'];order_position={f:i for i,f in enumerate(order)}
    donors=[parse(d['flow_block']) for d in payload['donors']];assert len(donors)==5
    baselines={};singles={};joints={};flow_done={}
    seen=set();executed_count=0;scored=[]
    for row in lines:
        event=row['event']
        assert event in {'starting','tokenization_verified','model_loaded','baseline','single','joint','flow_complete','complete','execution_start','execution_resume','execution_tokenization_verified','execution_model_loaded','execution_flow_reused','execution_progress'}
        if event in {'baseline','single','joint'}:
            fid=row['flow_id'];assert fid in targets
            assert isinstance(row['executed'],bool)
            # Phase-specific lineage and exact cache parity are verified by verify_chain_cache.
            seen.add(row['input_sha256']);executed_count+=int(row['executed']);scored.append(row)
        if event=='baseline':
            fid=row['flow_id'];assert fid not in baselines;baselines[fid]=row
            assert row['input_block']==targets[fid]['flow_block']
            assert row['input_sha256']==text_sha(row['input_block'])
            assert row['original_issues']==validity(parse(row['input_block']))
            q=row['q'];assert len(q)==15 and abs(sum(q.values())-1)<1e-10
            assert all(math.isfinite(v) and 0<=v<=1 for v in q.values())
            assert row['focus_class']==max(q,key=q.get)
        elif event=='single':
            key=(row['flow_id'],row['donor_index'],row['field']);assert key not in singles;singles[key]=row
        elif event=='joint':
            key=(row['flow_id'],row['donor_index'],row['method']);assert key not in joints;joints[key]=row
        elif event=='flow_complete':
            assert row['flow_id'] not in flow_done;flow_done[row['flow_id']]=row
    assert set(baselines)==set(flow_done)==set(targets)
    assert complete[0]['flows']==30
    assert complete[0]['actual_model_evaluations_total']==executed_count==cache_audit['actual_model_evaluations_total']
    assert complete[0]['distinct_model_evaluations']==len(seen)==cache_audit['distinct_input_hashes_evaluated_total']
    finishes=sorted(flow_done.values(),key=lambda r:r['ordinal'])
    assert [r['ordinal'] for r in finishes]==list(range(1,31))
    assert [r['flow_id'] for r in finishes]==[f['i'] for f in payload['flows']]
    assert all(r['total']==30 for r in finishes)
    assert finishes[-1]['distinct_model_evaluations']==len(seen)
    assert finishes[-1]['actual_model_evaluations_total']==executed_count
    planned_single=sum(len(parse(f['flow_block']))*len(donors) for f in targets.values())
    planned_joint=len(targets)*len(donors)*2
    assert len(singles)==planned_single and len(joints)==planned_joint
    flow_rows=[];intervention_rows=[];issues_by_kind=defaultdict(Counter)
    hash_focus_probs=defaultdict(list)
    for fid,flow in targets.items():
        baseline=baselines[fid];block=flow['flow_block'];values=parse(block)
        fields=[f for f in order if f in values];base_issues=set(baseline['original_issues'])
        assert flow_done[fid]['present_fields']==len(fields)
        focus=baseline['focus_class'];q0=float(baseline['q'][focus]);log0=math.log(max(q0,FLOOR))
        drop_by_donor=[];top_sets=[];top_ordered=[]
        for di,donor in enumerate(donors):
            ds={}
            for field in fields:
                row=singles[fid,di,field]
                assert row['donor_id']==payload['donors'][di]['donor_id']
                altered=intervene(values,donor,[field],order)
                assert row['input_sha256']==text_sha(altered)
                assert row['unchanged']==(altered==block)
                assert row['issues']==validity(parse(altered))
                assert math.isfinite(row['q_focus']) and 0<=row['q_focus']<=1
                expected=log0-math.log(max(row['q_focus'],FLOOR))
                assert math.isclose(expected,row['drop_log_probability'],rel_tol=1e-10,abs_tol=1e-10)
                if row['unchanged']:assert abs(row['drop_log_probability'])<1e-10
                ds[field]=row['drop_log_probability']
                enriched={**row,'unchanged':altered==block,'introduced_issues':sorted(set(row['issues'])-base_issues),
                          'baseline_issues':sorted(base_issues),'focus_class':focus,'baseline_q_focus':q0}
                intervention_rows.append(enriched)
                hash_focus_probs[row['input_sha256'],focus].append(row['q_focus'])
            drop_by_donor.append(ds)
            top=sorted(fields,key=lambda f:(-ds[f],order_position[f]))[:min(3,len(fields))]
            top_ordered.append(top);top_sets.append(set(top))
        jaccards=[];agreements=[];correlations=[];undefined=0
        for a,b in itertools.combinations(range(len(donors)),2):
            jaccards.append(len(top_sets[a]&top_sets[b])/len(top_sets[a]|top_sets[b]))
            agreements.append(float(top_sets[a]==top_sets[b]))
            corr=spearman_tied([drop_by_donor[a][f] for f in fields],[drop_by_donor[b][f] for f in fields])
            if corr is None:undefined+=1
            else:correlations.append(corr)
        k=min(3,len(fields));rng=np.random.default_rng(payload['random_control_seed']+fid)
        random_fields=list(rng.choice(fields,size=k,replace=False))
        pairs=[];eligible_pairs=[]
        for di,donor in enumerate(donors):
            ranking={f:float(np.mean([drop_by_donor[d][f] for d in range(len(donors)) if d!=di])) for f in fields}
            selected=sorted(fields,key=lambda f:(-ranking[f],order_position[f]))[:k]
            pair={};eligibility=True
            for method,selection in [('leave_one_donor_out',selected),('random',random_fields)]:
                row=joints[fid,di,method]
                assert row['selected_fields']==selection
                assert row['single_donor_top3']==top_ordered[di]
                assert row['donor_id']==payload['donors'][di]['donor_id']
                altered=intervene(values,donor,selection,order)
                assert row['input_sha256']==text_sha(altered)
                assert row['issues']==validity(parse(altered))
                assert math.isfinite(row['q_focus']) and 0<=row['q_focus']<=1
                assert math.isclose(log0-math.log(max(row['q_focus'],FLOOR)),row['drop_log_probability'],rel_tol=1e-10,abs_tol=1e-10)
                added=sorted(set(row['issues'])-base_issues)
                enriched={**row,'unchanged':altered==block,'introduced_issues':added,'baseline_issues':sorted(base_issues),
                          'focus_class':focus,'baseline_q_focus':q0}
                intervention_rows.append(enriched)
                hash_focus_probs[row['input_sha256'],focus].append(row['q_focus'])
                eligibility=eligibility and not added
                pair[method]=row['drop_log_probability']
            pair['difference']=pair['leave_one_donor_out']-pair['random'];pairs.append(pair)
            if eligibility:eligible_pairs.append(pair)
        flow_rows.append({'flow_id':fid,'source_row':flow['source_row'],'true':flow['true'],
            'input_sha256':baseline['input_sha256'],'focus_class':focus,'baseline_q_focus':q0,
            'present_fields':len(fields),'baseline_issues':sorted(base_issues),
            'pairwise_top3_jaccard_mean':float(np.mean(jaccards)),
            'pairwise_top3_exact_agreement_fraction':float(np.mean(agreements)),
            'all_five_top3_sets_identical':float(all(s==top_sets[0] for s in top_sets)),
            'pairwise_spearman_mean':float(np.mean(correlations)) if correlations else None,
            'pairwise_spearman_defined_pairs':len(correlations),'pairwise_spearman_undefined_pairs':undefined,
            'single_donor_top3_fields':top_ordered,'pairwise_jaccards':jaccards,'pairwise_spearman_defined':correlations,
            'lodo_joint_drop_mean':float(np.mean([x['leave_one_donor_out'] for x in pairs])),
            'random_joint_drop_mean':float(np.mean([x['random'] for x in pairs])),
            'paired_lodo_minus_random_mean':float(np.mean([x['difference'] for x in pairs])),
            'eligible_no_introduced_issue_pairs':len(eligible_pairs),
            'paired_no_introduced_issues_difference_mean':float(np.mean([x['difference'] for x in eligible_pairs])) if eligible_pairs else None,
            'lodo_no_introduced_issues_drop_mean':float(np.mean([x['leave_one_donor_out'] for x in eligible_pairs])) if eligible_pairs else None,
            'random_no_introduced_issues_drop_mean':float(np.mean([x['random'] for x in eligible_pairs])) if eligible_pairs else None})
    for values in hash_focus_probs.values():assert max(values)-min(values)<1e-10
    treatments={
        'single':[r for r in intervention_rows if r['event']=='single'],
        'joint_leave_one_donor_out':[r for r in intervention_rows if r['event']=='joint' and r['method']=='leave_one_donor_out'],
        'joint_random':[r for r in intervention_rows if r['event']=='joint' and r['method']=='random']}
    counts={}
    for kind,rr in treatments.items():
        counts[kind]={'planned_and_scored':len(rr),'fresh_model_evaluations':sum(r['executed'] for r in rr),
            'cached_events':sum(not r['executed'] for r in rr),'unchanged_events':sum(r['unchanged'] for r in rr),
            'changed_cached_events':sum(not r['executed'] and not r['unchanged'] for r in rr),
            'distinct_intervention_input_hashes':len({r['input_sha256'] for r in rr}),
            'distinct_intervention_input_hash_scope':'All planned intervention events, including unchanged substitutions and cached inputs.',
            'q_focus_below_floor':sum(r['q_focus']<FLOOR for r in rr),'q_focus_at_or_below_floor':sum(r['q_focus']<=FLOOR for r in rr),
            'minimum_raw_q_focus':min(r['q_focus'] for r in rr),'negative_drop_events':sum(r['drop_log_probability']<0 for r in rr),
            'events_with_any_flagged_issue':sum(bool(r['issues']) for r in rr),
            'events_with_newly_flagged_issue':sum(bool(r['introduced_issues']) for r in rr),
            'events_with_baseline_issue':sum(bool(r['baseline_issues']) for r in rr),
            'flagged_issue_code_counts':dict(Counter(c for r in rr for c in r['issues'])),
            'introduced_issue_code_counts':dict(Counter(c for r in rr for c in r['introduced_issues']))}
    old_finishes=[x for x in finishes if x.get('execution_phase')!='v3']
    new_finishes=[x for x in finishes if x.get('execution_phase')=='v3']
    assert len(old_finishes)==1 and old_finishes[0]['flow_id']==1 and len(new_finishes)==29
    elapsed=[];previous=0.
    for row in new_finishes:
        assert row['elapsed_scope']=='continuation' and row['elapsed_s']>=previous
        elapsed.append(row['elapsed_s']-previous);previous=row['elapsed_s']
    assert complete[0]['elapsed_scope']=='continuation' and complete[0]['elapsed_s']>=previous
    completion_minus_last_flow_elapsed_s=complete[0]['elapsed_s']-previous

    metrics=['pairwise_top3_jaccard_mean','pairwise_top3_exact_agreement_fraction','all_five_top3_sets_identical',
             'pairwise_spearman_mean','lodo_joint_drop_mean','random_joint_drop_mean','paired_lodo_minus_random_mean',
             'paired_no_introduced_issues_difference_mean','lodo_no_introduced_issues_drop_mean','random_no_introduced_issues_drop_mean']
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'Completed supplementary retrospective experiment; verified completion marker and successful launcher exit',
        'protocol_identifier':protocol['identifier'],'protocol_sha256':EXPECTED_PROTOCOL_SHA,'script_sha256':sha(Path(__file__)),
        'result_sha256':sha(result_path),'result_bytes':result_path.stat().st_size,
        'exit_verification_sha256':sha(args.exit_verification),'exit_verification':exit_evidence,
        'remote_pid':phase_start[0]['pid'],'original_prefix_remote_pid':start[0]['pid'],'boot_id':phase_start[0]['boot_id'],'model_sha256':protocol['model_sha256'],
        'execution_chain':{'amendment_sha256':EXPECTED_AMENDMENT_SHA,'immutable_prefix_sha256':amendment['immutable_v2_prefix_sha256'],'immutable_prefix_events':86,'immutable_prefix_bytes':len(prefix_bytes),'merged_prefix_byte_identical':True,'cache_and_counter_validation':cache_audit,'scientific_statistics_changed':False},
        'design':{'n_flow_occurrences':30,'n_unique_original_input_clusters':28,'input_cluster_multiplicities':dict(Counter(clusters.values())),
                  'n_classes':15,'flows_per_class':2,'n_fixed_validation_normal_donors':5,'held_out_donor_pairs_per_flow':5,
                  'n_planned_donor_pairs':150,'n_planned_pairwise_donor_comparisons':300,
                  'top3_tie_break':'Descending raw single-field log drop, then fixed source field order; exact numerical ties in Spearman receive average ranks.',
                  'ci_unit':'Original input hash clusters, preserving all 30 occurrence weights; fixed donor panel treated as repeated measurements, not independent flows.'},
        'completion_inventory':inventory(lines),'baselines':{'planned_and_scored':30,'fresh_model_evaluations':sum(r['executed'] for r in baselines.values()),
                 'cached_events':sum(not r['executed'] for r in baselines.values()),'distinct_input_hashes':28,
                 'q_focus_below_floor':sum(r['q'][r['focus_class']]<FLOOR for r in baselines.values()),
                 'all_class_q_below_floor':sum(v<FLOOR for r in baselines.values() for v in r['q'].values()),
                 'occurrences_with_flagged_issue':sum(bool(r['original_issues']) for r in baselines.values()),
                 'unique_input_clusters_with_flagged_issue':len({r['input_sha256'] for r in baselines.values() if r['original_issues']}),
                 'issue_code_counts':dict(Counter(c for r in baselines.values() for c in r['original_issues']))},
        'interventions':counts,'n_planned_single_events':planned_single,'n_planned_joint_events':planned_joint,
        'n_total_score_events_including_baselines':len(scored),'n_recorded_actual_model_evaluations':executed_count,'n_distinct_input_hashes_evaluated':len(seen),
        'mean_metrics':{m:cluster_summary(flow_rows,m) for m in metrics},
        'pairwise_spearman_defined_comparisons':sum(r['pairwise_spearman_defined_pairs'] for r in flow_rows),
        'pairwise_spearman_undefined_comparisons':sum(r['pairwise_spearman_undefined_pairs'] for r in flow_rows),
        'no_introduced_issue_subset':{'eligible_matched_donor_pairs':sum(r['eligible_no_introduced_issue_pairs'] for r in flow_rows),
            'eligible_flow_occurrences':sum(r['eligible_no_introduced_issue_pairs']>0 for r in flow_rows),
            'definition':'Both LODO and random joint replacements have no newly flagged issue relative to the same original baseline. Existing baseline issues may remain. Within each eligible flow, average only eligible donor-pair differences, then average flow occurrences with cluster resampling.'},
        'runtime':{'measurement':'Separate monotonic elapsed wall times of CPU-only original and continuation workers after model load/tokenization preflight. Not process CPU time or end-to-end decision latency.',
            'original_prefix_last_completed_flow_elapsed_s':old_finishes[-1]['elapsed_s'],
            'original_prefix_last_completed_flow_id':old_finishes[-1]['flow_id'],
            'continuation_worker_elapsed_s':complete[0]['elapsed_s'],
            'continuation_last_flow_complete_elapsed_s':new_finishes[-1]['elapsed_s'],
            'continuation_completion_minus_last_flow_elapsed_s':completion_minus_last_flow_elapsed_s,
            'completion_minus_last_flow_scope':'Descriptive continuation-worker gap from last flow_complete to complete; no upper-bound acceptance gate and no total or per-flow latency interpretation.',
            'continuation_flow_increment_elapsed_s':{'n_increment_events':29,'mean':float(np.mean(elapsed)),'median':float(np.median(elapsed)),'min':min(elapsed),'max':max(elapsed)},
            'total_full_experiment_elapsed_s':None,
            'elapsed_scope_limit':'Original prefix includes unfinished flow5 work after its last flow_complete; that time is not available in a completed-prefix timer. Do not add the clocks or treat continuation increments as complete per-flow decision latencies. Model-loading, interrupted unlogged work, optimization benchmark, SSH and downtime are excluded.',
            'thermal_c_range':[min(x['thermal_c'] for x in finishes),max(x['thermal_c'] for x in finishes)]},
        'limits':['New small class-balanced retained-record supplement; no verification of the missing historical 1499-flow intervention study.',
            'The 5 shared donors are a fixed panel. Cluster intervals describe target-input sampling conditional on that panel, not uncertainty across the full population of possible benign donors.',
            'Two observations per class and 28 distinct input clusters limit generalization; 150 donor pairs and 300 donor comparisons are not independent observations.',
            'Literal substitutions mix real donor fields with target fields. Limited port/flag/ACK checks do not establish physical or causal validity of mixed inputs.',
            'Probability floor 1e-12 censors extremely small scores; raw minimum probabilities and floor counts are retained. Negative drops remain included.',
            'Stable rank or larger log-drop is input sensitivity under this score definition, not proof of causal correctness or faithful model reasoning.',
            'Root blind-span review is an additional AI quality check; it does not create a second independent human evaluator or an inter-rater agreement measure.']}
    pp=dump_exclusive('pi_sensitivity_flow_analysis_v3.json',{'protocol_sha256':EXPECTED_PROTOCOL_SHA,'flows':flow_rows})
    ip=dump_exclusive('pi_sensitivity_intervention_analysis_v3.json',{'protocol_sha256':EXPECTED_PROTOCOL_SHA,'interventions':intervention_rows})
    sp=dump_exclusive('pi_sensitivity_analysis_v3.json',summary)
    dump_exclusive('pi_sensitivity_analysis_manifest_v3.json',{'created_utc':summary['created_utc'],'source_files_modified':False,
        'sources':{str(p):sha(p) for p in [result_path,old_path,ROOT/'INTERVENTION_PROTOCOL_v2.json',ROOT/'EXECUTION_CONTINUATION_v3.json',ROOT/'pi_sensitivity_payload_v2.json',ROOT/'pi_sensitivity_payload_v3.json',ROOT/'pi_sensitivity_remote_v2.py',ROOT/'pi_sensitivity_remote_v3.py',ROOT/'continuation_input_event_ledger_v3.json',args.exit_verification]},
        'outputs':[{'name':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in [pp,ip,sp]],'analysis_script_sha256':sha(Path(__file__))})
    print(json.dumps({'summary_file':str(sp),'sha256':sha(sp),'design':summary['design'],'baselines':summary['baselines'],
        'interventions':counts,'mean_metrics':summary['mean_metrics'],'no_introduced_issue_subset':summary['no_introduced_issue_subset'],
        'runtime':summary['runtime']},indent=2))

if __name__=='__main__':main()
