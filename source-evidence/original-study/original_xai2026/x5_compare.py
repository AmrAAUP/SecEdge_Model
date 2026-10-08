"""Track C (quantization vs XAI) and seed robustness.

1. Derive Q8 seed-456 records for the shared subsets (val_sub1500/test_sub1500 and
   val_sub300/test_sub300) from its full runs — same flows, same deterministic computation — so
   every model is calibrated on the same validation flows and scored on the same test flows.
2. Run the Track A analysis for each configuration.
3. Paired Q4-vs-Q8 comparison of conformal set sizes (C-T1, with the audit's McNemar test).
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

import xai_common as X

CONFIGS = [("q8_s456", "val_sub1500", "test_sub1500"), ("q8_s42", "val_sub1500", "test_sub1500"),
           ("q8_s123", "val_sub1500", "test_sub1500"), ("q4_s456", "val_sub1500", "test_sub1500"),
           ("q8_s456", "val_sub300", "test_sub300"), ("bf16_s456", "val_sub300", "test_sub300")]


def derive_q8_subsets():
    src = {"val": X.OUT / "x1_q8_s456_val_cal_dec.jsonl", "test": X.OUT / "x1_q8_s456_test_dec.jsonl"}
    recs = {k: {r["row"]: r for r in X.read_jsonl(p)} for k, p in src.items()}
    for sub in ("val_sub1500", "val_sub300", "test_sub1500", "test_sub300"):
        rows = pd.read_csv(X.OUT / "draws" / f"{sub}_index.csv")["row"].tolist()
        pool = recs["val" if sub.startswith("val") else "test"]
        with open(X.OUT / f"x1_q8_s456_{sub}_dec.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(pool[r], ensure_ascii=False) + "\n")


def run_trackA():
    for model, val, test in CONFIGS:
        out = X.OUT / f"x2_trackA_{model}_{val}.json"
        if out.exists():
            continue
        X.log(f"Track A: {model} {val} -> {test}")
        subprocess.run([sys.executable, str(X.ROOT / "x2_trackA.py"), "--model", model, "--val", val,
                        "--tests", test, "--boot", "1000"], check=True)


def summarize():
    rows = {}
    for model, val, test in CONFIGS:
        d = json.load(open(X.OUT / f"x2_trackA_{model}_{val}.json"))
        t = d["tests"][test]
        pc, pr = d["primary_calibrator"], d["primary_ranking"]
        c = t["conformal"]["lac_mondrian_0.05"]
        rows[f"{model}|{test}"] = {
            "accuracy": t["accuracy"], "macro_f1": t["macro_f1"],
            "verbal_ece": t["methods"]["verbal_raw"]["ece"], "verbal_auroc": t["methods"]["verbal_raw"]["auroc"],
            "primary_calibrator": pc, "primary_ece": t["methods"][pc]["ece"],
            "primary_ranking": pr, "primary_auroc": t["methods"][pr]["auroc"],
            "primary_aurc": t["methods"][pr]["aurc"], "temperature": d["temperature"],
            "conformal_coverage": c["coverage"], "conformal_mean_size": c["mean_size"],
            "singleton_rate": c["singleton_rate"], "all_classes_ok": c["all_classes_ok"]}
    # paired set-size difference Q4 - Q8 on the same test_sub1500 flows
    a = np.load(X.OUT / "x2_q8_s456_test_sub1500_scores.npz")
    b = np.load(X.OUT / "x2_q4_s456_test_sub1500_scores.npz")
    sa, sb = a["sets"].sum(1), b["sets"].sum(1)
    diff = sb - sa
    lo, hi = X.bootstrap_ci(np.mean, diff.astype(float), n_boot=10000)
    from scipy.stats import wilcoxon
    p = float(wilcoxon(sb, sa, zero_method="wilcox").pvalue) if np.any(diff) else 1.0
    audit = json.load(open(X.OUT / "x3_audit_q8_s456.json"))
    paired_unobs = audit.get("paired", {}).get("unobservable", {})
    q4u = audit.get("compare", {}).get("summary", {}).get("unobservable_claim", {}).get("rate")
    q8u = audit["summary"]["unobservable_claim"]["rate"]
    ct1 = {"mean_set_size_q8": float(sa.mean()), "mean_set_size_q4": float(sb.mean()),
           "paired_diff_q4_minus_q8": float(diff.mean()), "diff_ci95": [lo, hi], "wilcoxon_p": p,
           "unobservable_rate_q8": q8u, "unobservable_rate_q4": q4u, "unobservable_mcnemar": paired_unobs}
    ct1["pass"] = bool(diff.mean() > 0 and lo > 0 and q4u is not None and q4u > q8u
                       and paired_unobs.get("p_exact", 1) < 0.05)
    seeds = [rows[f"q8_s{s}|test_sub1500"] for s in (42, 123, 456)]
    robust = {k: {"mean": float(np.mean([s[k] for s in seeds])), "sd": float(np.std([s[k] for s in seeds], ddof=1))}
              for k in ("accuracy", "verbal_ece", "primary_ece", "primary_auroc", "primary_aurc",
                        "conformal_coverage", "conformal_mean_size", "singleton_rate")}
    out = {"configs": rows, "C-T1": ct1, "seed_robustness_q8": robust}
    json.dump(out, open(X.OUT / "x5_trackC.json", "w"), indent=1)
    print(json.dumps(out["C-T1"], indent=1))
    print(json.dumps(robust, indent=1))


if __name__ == "__main__":
    derive_q8_subsets()
    run_trackA()
    summarize()
