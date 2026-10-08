"""Track B2/B3 — counterfactual field attribution, faithfulness of explanations, evidence-bound
explanations; plus the GPU reference used by Track D.

run      (GPU) 1) GPU reference for the 150 Pi flows (deployed decision + canonical-prefix dist)
               2) for 750 flows (50/class of test_sub1500): single-field in-distribution
                  interventions and joint interventions for four evidence rankings
analyze  (CPU) comprehensiveness / sufficiency, SLM-vs-SHAP agreement, evidence-bound explanation
               text + its audit and cost.

An intervention replaces a field's value with a value drawn from BENIGN (Normal) training flows in
which that field is also present, so the rendered block keeps its presence signature and stays
inside the instruction-tuning distribution.
"""
import argparse
import json
import re
import time
import zlib

import numpy as np
import pandas as pd
import xgboost as xgb

import xai_common as X
import x3_audit as AUD

FIELDS = X.CFG["flow_block"]["field_order"]            # [label, column, rule]
LAB2COL = {lab: col for lab, col, _ in FIELDS}
N_PER_CLASS = 50
K_DRAWS = 2
POOLS = X.OUT / "normal_pools.json"


# ------------------------------------------------------------------ pools of benign values
def normal_pools(size=20000) -> dict:
    if POOLS.exists():
        return json.load(open(POOLS, encoding="utf-8"))
    cols = [c for _, c, _ in FIELDS] + ["Attack_type"]
    tr = pd.read_csv(X.SPLITS / "train.csv", usecols=lambda c: c in cols, low_memory=False)
    tr = tr[tr["Attack_type"].astype(str) == "Normal"]
    pools = {}
    for lab, col, rule in FIELDS:
        s = tr[col]
        t = s.astype(str).str.strip()
        ok = s.notna() & (t != "") & (t.str.lower() != "nan")
        if rule == "nonzero":
            num = pd.to_numeric(t, errors="coerce")
            ok &= ~(num.notna() & (num == 0.0))
        v = s[ok]
        pools[lab] = [str(x) for x in v.sample(min(size, len(v)), random_state=2026)] if len(v) else []
    json.dump(pools, open(POOLS, "w", encoding="utf-8"))
    return pools


# ------------------------------------------------------------------ rationale → cited fields
KEYWORDS = [
    (r"\bURI\b|\bURL\b|\bquery\b|\bparameter", ["http_full_uri", "http_uri_query"]),
    (r"\bGET\b|\bPOST\b|\bmethod\b|\bHTTP\b|\bweb request\b", ["http_method", "http_full_uri"]),
    (r"\bMQTT\b|\btopic\b|\bpublish", ["mqtt_topic", "mqtt_msgtype"]),
    (r"\bDNS\b", ["dns_qry_name"]),
    (r"\bARP\b", ["arp_opcode"]),
    (r"\bSYN\b|\bACK\b|\bRST\b|\bFIN\b|\bflags?\b|\bhandshake\b", ["tcp_flags", "tcp_flags_ack"]),
    (r"\blength\b|\bpayload size\b|\bsegment", ["tcp_len"]),
]


def cited_fields(rationale: str, block: dict) -> list[str]:
    text = rationale or ""
    hits = []  # (position, field)
    for ip in AUD.RE_IP.finditer(text):
        for f in ("src_host", "dst_host"):
            if block.get(f) == ip.group(0):
                hits.append((ip.start(), f))
    for m in AUD.RE_PORT.finditer(text):
        p = AUD._num(m.group(1))
        for f in ("tcp_dstport", "tcp_srcport"):
            if f in block and AUD._num(block[f]) == p:
                hits.append((m.start(), f))
    for pat, fs in KEYWORDS:
        m = re.search(pat, text, re.I)
        if m:
            hits += [(m.start(), f) for f in fs if f in block]
    seen, out = set(), []
    for _, f in sorted(hits):
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# ------------------------------------------------------------------ scoring helpers
class Scorer:
    def __init__(self, srv, trie):
        self.srv, self.trie, self.cache, self.calls = srv, trie, {}, 0

    def logp(self, row: pd.Series, target: str) -> float:
        fb = X.L.build_flow_block(row, X.CFG)
        if fb not in self.cache:
            ids = X.encode(X.L.build_prompt(fb, X.CFG))
            self.cache[fb], _ = X.class_distribution(self.srv, self.trie, ids + self.trie.prefix_ids)
            self.calls += 1
        return float(np.log(max(self.cache[fb][target], 1e-12)))


def replaced(row: pd.Series, fields: list[str], pools: dict, rng_seed: int, draw: int) -> pd.Series:
    r = row.copy()
    for f in fields:
        pool = pools.get(f) or []
        if not pool:
            continue
        # stable seed (Python's hash() of strings is randomised per process)
        rng = np.random.default_rng(zlib.crc32(f"{rng_seed}|{f}|{draw}".encode()))
        orig = str(row[LAB2COL[f]]).strip()
        for _ in range(10):
            v = pool[int(rng.integers(0, len(pool)))]
            if v.strip() != orig:
                break
        r[LAB2COL[f]] = v
    return r


def shap_for(gate, df: pd.DataFrame, labels: list[str]) -> list[dict]:
    inner = gate.model.model
    bst = inner.get_booster()
    Xm = X.L.to_feature_matrix(df, gate.feature_cols)
    C = np.asarray(bst.predict(xgb.DMatrix(Xm), pred_contribs=True))  # (n, K, F+1) internal order
    order = gate.model.column_order
    out = []
    for i, lab in enumerate(labels):
        k_int = order[X.CLASSES.index(lab)] if lab in X.CLASSES else 0
        vals = C[i, k_int, :-1]
        out.append({col: float(vals[j]) for j, col in enumerate(gate.feature_cols)})
    return out


# ------------------------------------------------------------------ run (GPU)
def gpuref(srv, trie, test):
    B = json.load(open(X.OUT / "pi_bundle.json"))
    log = X.Resumable("x6_gpu_reference")
    for f in B["flows"]:
        if log.has(f["row"]):
            continue
        row = test.loc[f["row"]]
        ids = X.encode(X.prompt_for_row(row))
        dec = X.decide(srv, ids)
        dist_c, _ = X.class_distribution(srv, trie, ids + trie.prefix_ids)
        log.write(f["row"], {"row": f["row"], "true": f["true"], "label": dec["label"], "conf": dec["conf"],
                             "dist_canonical": dist_c})
    log.close()


def run():
    test = pd.read_csv(X.SAMPLES / "largesample_n6184.csv", low_memory=False)
    full = {r["row"]: r for r in X.read_jsonl(X.OUT / "x1_q8_s456_test_sub1500_full.jsonl")}
    sub = pd.read_csv(X.OUT / "draws" / "test_sub1500_index.csv")
    rows = []
    for c in X.CLASSES:
        rows += sub[sub["Attack_type"] == c]["row"].head(N_PER_CLASS).tolist()
    pools = normal_pools()
    gate = X.L.Stage1Gate.load(X.CFG)
    trie = X.ClassTrie(X.prompt_for_row(test.loc[rows[0]]))
    X.verify_model("q8_s456")
    srv = X.Server("q8_s456", port=8091)
    srv.start(timeout=900)
    log = X.Resumable("x4_counterfactual")
    try:
        gpuref(srv, trie, test)
        X.log("GPU reference for Track D done")
        labels = [full[r]["label"] if r in full else None for r in rows]
        shaps = dict(zip(rows, shap_for(gate, test.loc[rows], [l or "Normal" for l in labels])))
        sc = Scorer(srv, trie)
        t0 = time.time()
        for n, r in enumerate(rows, 1):
            if log.has(r) or r not in full:
                continue
            row, rec = test.loc[r], full[r]
            yhat = rec["label"]
            block = AUD.parse_block(rec["flow_block"])
            present = [lab for lab, _, _ in FIELDS if lab in block]
            base = sc.logp(row, yhat)
            single = {}
            for f in present:
                single[f] = float(np.mean([base - sc.logp(replaced(row, [f], pools, r, d), yhat)
                                           for d in range(K_DRAWS)]))
            rng = np.random.default_rng(r)
            rand = list(rng.permutation(present))
            cited = cited_fields(rec.get("rationale"), block)
            sh = shaps[r]
            shap_rel = sorted([f for f in present if LAB2COL[f] in sh], key=lambda f: -abs(sh[LAB2COL[f]]))
            rankings = {
                "cf": sorted(present, key=lambda f: -single[f]),
                "rat": cited + [f for f in rand if f not in cited],
                "shap": shap_rel + [f for f in rand if f not in shap_rel],
                "rand": rand,
            }
            comp, suff = {}, {}
            for name, R in rankings.items():
                for k in (1, 3):
                    comp[f"{name}@{k}"] = base - sc.logp(replaced(row, R[:k], pools, r, 0), yhat)
                keep = set(R[:3])
                suff[f"{name}@3"] = base - sc.logp(replaced(row, [f for f in present if f not in keep],
                                                            pools, r, 0), yhat)
            comp["rat_cited_only@3"] = (base - sc.logp(replaced(row, cited[:3], pools, r, 0), yhat)
                                        if cited else 0.0)
            log.write(r, {"row": r, "true": rec["true"], "label": yhat, "correct": yhat == rec["true"],
                          "present": present, "base_logp": base, "single": single,
                          "rankings": rankings, "cited": cited, "n_cited": len(cited),
                          "shap": {f: sh[LAB2COL[f]] for f in shap_rel}, "comp": comp, "suff": suff})
            if n % 25 == 0:
                el = time.time() - t0
                X.log(f"  counterfactual {n}/{len(rows)}  {el / n:.1f} s/flow  calls={sc.calls}")
    finally:
        srv.stop()
        log.close()


# ------------------------------------------------------------------ analyze (CPU)
def technique_names() -> dict:
    stix = json.load(open(X.BUNDLE / "artifacts" / "enterprise-attack.json", encoding="utf-8"))
    names = {}
    for o in stix.get("objects", []):
        if o.get("type") == "attack-pattern":
            for ref in o.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    names[ref.get("external_id")] = o.get("name")
    return names


def evidence_text(label: str, fields: list[str], block: dict, names: dict, cset: list[str] | None) -> str:
    tid = AUD.MITRE.get(label)
    head = f"{label}" + (f" (ATT&CK {tid}, {names.get(tid, 'technique')})" if tid else " (no attack technique)")
    ev = "; ".join(f"{f}={block[f]}" for f in fields if f in block)
    s = f"{head}. Decisive evidence in this flow: {ev}."
    if cset and len(cset) > 1:
        others = [c for c in cset if c != label]
        # phrased without a numeral: the confidence level is a system setting, not a flow value
        s += f" Not certain: at the calibrated confidence level the model cannot exclude {', '.join(others)}."
    return s


def analyze():
    recs = X.read_jsonl(X.OUT / "x4_counterfactual.jsonl")
    names = technique_names()
    npz = np.load(X.OUT / "x2_q8_s456_test_scores.npz")
    test_idx = pd.read_csv(X.OUT / "draws" / "test_index.csv")
    pos = {int(r): i for i, r in enumerate(test_idx["row"])}
    sets = npz["sets"]
    full = {r["row"]: r for r in X.read_jsonl(X.OUT / "x1_q8_s456_test_sub1500_full.jsonl")}

    def mean_ci(v):
        v = np.asarray(v, float)
        lo, hi = X.bootstrap_ci(np.mean, v, n_boot=2000)
        return {"mean": float(v.mean()), "ci95": [lo, hi], "n": int(len(v))}

    out = {"n_flows": len(recs), "comprehensiveness": {}, "sufficiency": {}}
    for key in ["cf@1", "cf@3", "rat@1", "rat@3", "shap@1", "shap@3", "rand@1", "rand@3", "rat_cited_only@3"]:
        out["comprehensiveness"][key] = mean_ci([r["comp"][key] for r in recs])
    for key in ["cf@3", "rat@3", "shap@3", "rand@3"]:
        out["sufficiency"][key] = mean_ci([r["suff"][key] for r in recs])
    c3 = out["comprehensiveness"]
    out["B-T2"] = {"cf@3": c3["cf@3"]["mean"], "rand@3": c3["rand@3"]["mean"],
                   "ratio": c3["cf@3"]["mean"] / max(c3["rand@3"]["mean"], 1e-9)}
    out["B-T2"]["pass"] = out["B-T2"]["ratio"] >= 2.0
    # SLM-vs-SHAP agreement
    ov_fill, ov_rel, taus = [], [], []
    from scipy.stats import kendalltau
    for r in recs:
        k = min(3, len(r["present"]))
        if k == 0:
            continue
        ov_fill.append(len(set(r["rankings"]["cf"][:k]) & set(r["rankings"]["shap"][:k])) / k)
        rel = list(r["shap"].keys())
        if len(rel) >= 2:
            cf_rel = [f for f in r["rankings"]["cf"] if f in rel][:min(3, len(rel))]
            ov_rel.append(len(set(cf_rel) & set(rel[:len(cf_rel)])) / len(cf_rel))
            t, _ = kendalltau([r["single"][f] for f in rel], [abs(r["shap"][f]) for f in rel])
            if not np.isnan(t):
                taus.append(t)
    out["B-T4"] = {"top3_overlap_with_fill": float(np.mean(ov_fill)),
                   "top3_overlap_shap_relevant_only": float(np.mean(ov_rel)) if ov_rel else None,
                   "kendall_tau_mean": float(np.mean(taus)) if taus else None,
                   "share_flows_top_cf_field_visible_to_gate": float(np.mean(
                       [r["rankings"]["cf"][0] in r["shap"] for r in recs if r["present"]]))}
    out["B-T4"]["pass"] = out["B-T4"]["top3_overlap_with_fill"] >= 0.6
    # evidence-bound explanation (GPU-attribution variant = cf; on-device variant = shap)
    ex_rows, t_gen = [], []
    for r in recs:
        block = AUD.parse_block(full[r["row"]]["flow_block"])
        s = sets[pos[r["row"]]]
        cset = [X.CLASSES[j] for j in np.where(s)[0]]
        t0 = time.perf_counter()
        txt = evidence_text(r["label"], r["rankings"]["cf"][:3], block, names, cset)
        t_gen.append(time.perf_counter() - t0)
        a = AUD.audit_text(txt, block)
        ex_rows.append({"row": r["row"], "text": txt, "hallucinated": a["has_hallucination"],
                        "unobservable": a["has_unobservable"], "n_fields": min(3, len(r["present"]))})
    rat_only = c3["rat_cited_only@3"]["mean"]
    out["B-T3"] = {"evidence_bound_comp@3": c3["cf@3"]["mean"], "rationale_cited_only_comp@3": rat_only,
                   "ratio": c3["cf@3"]["mean"] / max(rat_only, 1e-9),
                   "hallucinated": int(sum(x["hallucinated"] for x in ex_rows)),
                   "unobservable": int(sum(x["unobservable"] for x in ex_rows)),
                   "mean_fields_cited_by_rationale": float(np.mean([r["n_cited"] for r in recs])),
                   "share_rationales_citing_no_present_field": float(np.mean([r["n_cited"] == 0 for r in recs])),
                   "text_generation_ms_mean": 1000 * float(np.mean(t_gen))}
    out["B-T3"]["pass"] = (out["B-T3"]["ratio"] >= 1.5 and out["B-T3"]["hallucinated"] == 0
                           and out["B-T3"]["unobservable"] == 0)
    # SHAP cost on this machine
    gate = X.L.Stage1Gate.load(X.CFG)
    test = pd.read_csv(X.SAMPLES / "largesample_n6184.csv", low_memory=False)
    one = test.loc[[recs[0]["row"]]]
    t0 = time.perf_counter()
    for _ in range(50):
        shap_for(gate, one, [recs[0]["label"]])
    out["B-T4"]["shap_ms_per_flow_workstation"] = 1000 * (time.perf_counter() - t0) / 50
    # per-class view of which field decides
    out["top_counterfactual_field_by_class"] = {}
    for c in X.CLASSES:
        tops = [r["rankings"]["cf"][0] for r in recs if r["true"] == c and r["present"]]
        if tops:
            vc = pd.Series(tops).value_counts(normalize=True)
            out["top_counterfactual_field_by_class"][c] = {k: round(float(v), 3) for k, v in vc.head(3).items()}
    json.dump(out, open(X.OUT / "x4_trackB.json", "w"), indent=1)
    with open(X.OUT / "x4_evidence_explanations.jsonl", "w", encoding="utf-8") as fh:
        for x in ex_rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("B-T2", "B-T3", "B-T4")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze", "pools"])
    cmd = ap.parse_args().cmd
    if cmd == "pools":
        p = normal_pools()
        print({k: len(v) for k, v in p.items()})
    else:
        {"run": run, "analyze": analyze}[cmd]()
