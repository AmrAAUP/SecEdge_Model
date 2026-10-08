"""Independent re-derivation of every headline number in RESULTS.json from the per-flow records.

Prints OK / MISMATCH per claim; exits non-zero on any mismatch.
"""
import json
import sys

import numpy as np

import xai_common as X
import x2_trackA as A
import x3_audit as AUD

R = json.load(open(X.ROOT / "RESULTS.json", encoding="utf-8"))
bad = 0


def check(name, got, want, tol=1e-6):
    global bad
    ok = (got is None and want is None) or (got is not None and want is not None and abs(got - want) <= tol)
    bad += 0 if ok else 1
    print(f"{'OK      ' if ok else 'MISMATCH'} {name}: recomputed {got} | reported {want}")


# ---------------- Track A: refit from raw validation records, rescore raw test records
ta = R["trackA"]
fv = A.feats(A.load("q8_s456", "val_cal"))
ft = A.feats(A.load("q8_s456", "test"))
check("test accuracy", float(ft["correct"].mean()), ta["tests"]["test"]["accuracy"])
check("temperature", A.fit_T(fv["P"], fv["y"]), ta["temperature"], tol=1e-4)
for kind in (ta["primary_calibrator"], ta["primary_ranking"], "verbal_raw"):
    s = np.clip(A.Calib(kind).fit(fv).predict(ft), 0, 1)
    m = ta["tests"]["test"]["methods"][kind]
    check(f"{kind} ECE", X.ece(s, ft["correct"])[0], m["ece"])
    check(f"{kind} AUROC", X.auroc(s, ft["correct"]), m["auroc"])
    check(f"{kind} AURC", X.aurc(s, ft["correct"])[0], m["aurc"])
Pv, Pt = A.apply_T(fv["P"], ta["temperature"]), A.apply_T(ft["P"], ta["temperature"])
r, sets = A.conformal(Pv, fv["y"], Pt, ft["y"], "lac", 0.05, True)
c = ta["tests"]["test"]["conformal"]["lac_mondrian_0.05"]
check("conformal coverage", r["coverage"], c["coverage"])
check("conformal mean size", r["mean_size"], c["mean_size"])
check("conformal singleton rate", r["singleton_rate"], c["singleton_rate"])

# ---------------- Track B1: audit rates from the stored rows
rows = [x for x in X.read_jsonl(X.OUT / "x3_audit_q8_s456_rows.jsonl") if x["parsed"]]
au = R["audit"]["summary"]
check("hallucinated rate", float(np.mean([x["has_hallucination"] for x in rows])), au["hallucinated_value"]["rate"])
check("unobservable rate", float(np.mean([x["has_unobservable"] for x in rows])), au["unobservable_claim"]["rate"])
# re-audit the raw rationales (catches drift between stored rows and the auditor)
full = {x["row"]: x for x in X.read_jsonl(X.OUT / "x1_q8_s456_test_sub1500_full.jsonl")}
re_unobs = [AUD.audit_text(full[x["row"]]["rationale"], AUD.parse_block(full[x["row"]]["flow_block"]))["has_unobservable"]
            for x in rows]
check("unobservable rate (re-audited from raw)", float(np.mean(re_unobs)), au["unobservable_claim"]["rate"])

# ---------------- Track B2/B3: comprehensiveness from the per-flow interventions
cf = X.read_jsonl(X.OUT / "x4_counterfactual.jsonl")
tb = R["trackB"]
for k in ("cf@3", "rand@3", "rat_cited_only@3"):
    check(f"comprehensiveness {k}", float(np.mean([x["comp"][k] for x in cf])), tb["comprehensiveness"][k]["mean"])

# ---------------- Track D
td = R["trackD"]
import x6_pi as P
B = json.load(open(P.BUNDLE))
pi = {x["row"]: x for x in X.read_jsonl(P.RESULTS)}
ref = {x["row"]: x for x in X.read_jsonl(X.OUT / "x6_gpu_reference.jsonl")}
agree = [X.parse_decision(pi[f["row"]]["text"])[0] == ref[f["row"]]["label"] for f in B["flows"]
         if f["row"] in pi and f["row"] in ref]
check("GPU-Pi label agreement", float(np.mean(agree)), td["label_agreement_gpu_pi"])

print(f"\n{'ALL CLAIMS VERIFIED' if bad == 0 else f'{bad} MISMATCH(ES)'}")
sys.exit(1 if bad else 0)
