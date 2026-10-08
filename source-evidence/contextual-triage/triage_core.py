"""Authored demonstration policy; no model inference or control actions."""
import math
CLASSES={'Normal','DDoS_TCP','DDoS_UDP','DDoS_ICMP','DDoS_HTTP','SQL_injection','Password','Vulnerability_scanner','Uploading','Backdoor','Port_Scanning','XSS','Ransomware','MITM','Fingerprinting'}

def triage(record, profile, mode):
    label=record.get('label')
    label_valid=label in CLASSES
    conf=record.get('confidence')
    conf_valid=isinstance(conf,(int,float)) and not isinstance(conf,bool) and math.isfinite(conf) and 0<=conf<=1
    floor_review=mode!='label_only' and (not conf_valid or conf<0.50)
    candidates={label} if label_valid else set()
    set_issue=False
    if mode=='set_augmented':
        labels=record.get('labels_in_set')
        set_issue=not isinstance(labels,list) or not labels or any(x not in CLASSES for x in labels)
        if isinstance(labels,list):candidates.update(x for x in labels if x in CLASSES)
    elif mode not in ('label_only','confidence_floor'):
        raise ValueError(mode)
    model_review=not label_valid or floor_review or set_issue
    if profile['serious_impact_reported'] is True:
        return ('High',True,'reported_serious_impact')
    if not profile['context_known']:
        return ('Unassigned',True,'context_review')
    if candidates-{'Normal'}:
        return ('Moderate' if profile['critical_asset'] else 'Low',True,'attack_candidate_and_model_review' if model_review else 'attack_candidate')
    if model_review or not candidates:
        return ('Unassigned',True,'model_review')
    return ('Not_indicated',False,'routine')
