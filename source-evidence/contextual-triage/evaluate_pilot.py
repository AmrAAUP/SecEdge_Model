from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,itertools
from triage_core import triage,CLASSES

R=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
protocol=json.loads((R/'PROTOCOL.json').read_text(encoding='utf-8'))
assert sha((R/'PROTOCOL.json').read_bytes())==(R/'PROTOCOL.sha256').read_text().split()[0]
records=json.loads((R/'input_records.json').read_text(encoding='utf-8'))
if isinstance(records,dict):records=records['records']
profiles=protocol['profiles'];modes=list(protocol['modes'])
summary={};counter=Counter();decisions=[]
for pop in sorted({r['population'] for r in records}):
    rr=[r for r in records if r['population']==pop]
    assert len({r['row'] for r in rr})==len(rr)
    truth_attack=sum(r['true']!='Normal' for r in rr);truth_normal=len(rr)-truth_attack
    out={}
    for p in profiles:
        pp={}
        for mode in modes:
            vals=[triage(r,p,mode) for r in rr]
            false_routine=sum(r['true']!='Normal' and not v[1] for r,v in zip(rr,vals))
            normal_review=sum(r['true']=='Normal' and v[1] for r,v in zip(rr,vals))
            review=sum(v[1] for v in vals)
            pp[mode]={'n':len(rr),'true_attacks':truth_attack,'true_normal':truth_normal,
              'attacks_routed_routine':false_routine,'attacks_routed_routine_rate':false_routine/truth_attack,
              'normal_routed_review':normal_review,'normal_routed_review_rate':normal_review/truth_normal,
              'all_review':review,'all_review_rate':review/len(rr),
              'severity_counts':dict(Counter(v[0] for v in vals)),
              'reason_counts':dict(Counter(v[2] for v in vals))}
            for rec,v in zip(rr,vals):
                decisions.append({'population':pop,'row':rec['row'],'true':rec['true'],'profile':p['id'],'mode':mode,'severity':v[0],'review_required':v[1],'reason':v[2],'flow_sha256':rec['flow_sha256']})
        base=pp['confidence_floor'];new=pp['set_augmented']
        pp['main_comparison']={'additional_attack_referrals':base['attacks_routed_routine']-new['attacks_routed_routine'],'additional_normal_referrals':new['normal_routed_review']-base['normal_routed_review'],'additional_total_referrals':new['all_review']-base['all_review']}
        out[p['id']]=pp
    summary[pop]={'n':len(rr),'unique_rendered_inputs':len({r['flow_sha256']for r in rr}),'profiles':out}
checks=[]
def check(name,passed,count):
    assert passed,name
    checks.append({'property':name,'pass':passed,'cases':count,'interpretation':'Deterministic implementation property, not empirical severity validation.'})
all_positive=True;mono=True;expand=True;high=True;unknown=True;empty=True;count=0
ordinal={'Not_indicated':0,'Low':1,'Moderate':2,'High':3}
for label,conf in itertools.product(sorted(CLASSES),[0.1,0.49,0.5,0.99,None]):
    rr={'label':label,'confidence':conf,'labels_in_set':[label]};count+=1
    for mode in modes:
        a=triage(rr,profiles[0],mode);b=triage(rr,profiles[1],mode)
        mono &= (a[0] not in ordinal or b[0] not in ordinal or ordinal[b[0]]>=ordinal[a[0]])
        high &= triage(rr,profiles[2],mode)==('High',True,'reported_serious_impact')
        unknown &= triage(rr,profiles[3],mode)==('Unassigned',True,'context_review')
        if label!='Normal':all_positive &= a[0]=='Low' and b[0]=='Moderate'
    for extra in sorted(CLASSES):
        e=dict(rr,labels_in_set=list({label,extra}))
        for p in profiles:
            expand &= not triage(rr,p,'set_augmented')[1] or triage(e,p,'set_augmented')[1]
    for p in profiles:empty &= triage(dict(rr,labels_in_set=[]),p,'set_augmented')[1]
check('Known criticality never reduces assigned severity',bool(mono),count*3)
check('Supplied serious impact overrides model outputs',bool(high),count*3)
check('Unknown context always requests context review',bool(unknown),count*3)
check('Prediction-set expansion never suppresses a referral',bool(expand),count*15*4)
check('Empty prediction set requests review',bool(empty),count*4)
check('Confidence never changes positive Low/Moderate tiers',bool(all_positive),count)
raw=('\n'.join(json.dumps(x,sort_keys=True,separators=(',',':'))for x in decisions)+'\n').encode('utf-8')
(R/'decision_ledger.jsonl').write_bytes(raw)
result={'experiment':'Exploratory contextual severity triage pilot','protocol_sha256':sha((R/'PROTOCOL.json').read_bytes()),'input_sha256':sha((R/'input_records.json').read_bytes()),'core_sha256':sha((R/'triage_core.py').read_bytes()),'ledger_sha256':sha(raw),'ledger_rows':len(decisions),'summary':summary,'implementation_checks':checks,'interpretation':protocol['metric_interpretation']}
(R/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({pop:row['profiles']['P1_noncritical'] for pop,row in summary.items()},indent=2))
