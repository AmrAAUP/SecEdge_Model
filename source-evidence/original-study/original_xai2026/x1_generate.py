"""Phase 1 — per-flow generation for every track (resumable, GPU).

Draws (all deterministic, written to out/draws/ with SHA-256):
  val_cal      validation split, stratified, <= CAP_VAL per class (seed 2026) — calibration only
  val_sub1500  100/class subset of val_cal — per-model calibration for seeds / Q4
  val_sub300   20/class subset of val_cal  — bf16 calibration
  test         frozen pre-registered largesample_n6184 draw (never used for fitting)
  realistic    frozen realistic_n600 draw (deployment class mix)
  test_sub1500 100/class subset of test (seed 2026)
  test_sub300  20/class subset of test_sub1500

Jobs (run in order with --jobs, each resumable):
  A  q8_s456   decision+trie   val_cal, test, realistic
  R  q8_s456   full schema     test_sub1500                 (Track B rationales)
  S  q8_s42/123 decision+trie  val_sub1500, test_sub1500    (seed robustness)
  Q  q4_s456   decision+trie   val_sub1500, test_sub1500; full schema test_sub1500
  F  bf16_s456 decision+trie   val_sub300, test_sub300      (partial GPU offload)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import xai_common as X

CAP_VAL = 250
SEED = 2026
DRAWS = X.OUT / "draws"
DRAWS.mkdir(exist_ok=True)


def _stratified(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for c in X.CLASSES:
        g = df[df["Attack_type"].astype(str) == c]
        if len(g):
            parts.append(g.sample(min(per_class, len(g)), random_state=seed))
    return pd.concat(parts).sort_index()


def build_draws() -> dict[str, pd.DataFrame]:
    out = {}
    cache = DRAWS / "val_cal.csv"
    if cache.exists():
        out["val_cal"] = pd.read_csv(cache, low_memory=False, index_col=0)
    else:
        X.log("reading val split ...")
        val = pd.read_csv(X.SPLITS / "val.csv", low_memory=False)
        out["val_cal"] = _stratified(val, CAP_VAL, SEED)
        out["val_cal"].to_csv(cache)
        del val
    out["test"] = pd.read_csv(X.SAMPLES / "largesample_n6184.csv", low_memory=False)
    out["realistic"] = pd.read_csv(X.SAMPLES / "realistic_n600.csv", low_memory=False)
    out["val_sub1500"] = _stratified(out["val_cal"], 100, SEED)
    out["val_sub300"] = _stratified(out["val_sub1500"], 20, SEED)
    out["test_sub1500"] = _stratified(out["test"], 100, SEED)
    out["test_sub300"] = _stratified(out["test_sub1500"], 20, SEED)
    manifest = {}
    for name, df in out.items():
        p = DRAWS / f"{name}_index.csv"
        pd.DataFrame({"row": df.index, "Attack_type": df["Attack_type"].astype(str)}).to_csv(p, index=False)
        manifest[name] = {"n": int(len(df)), "per_class": df["Attack_type"].value_counts().to_dict(),
                          "index_sha256": X.sha256_file(p)}
    json.dump(manifest, open(DRAWS / "manifest.json", "w"), indent=1)
    return out


def gate_probs(df: pd.DataFrame) -> list[dict]:
    gate = X.L.Stage1Gate.load(X.CFG)
    P = gate.predict_proba(df)
    return [{c: float(P[i, j]) for j, c in enumerate(gate.classes)} for i in range(len(df))]


def run(model: str, draw: str, df: pd.DataFrame, full: bool, trie_on: bool, srv, trie) -> None:
    tag = f"x1_{model}_{draw}_{'full' if full else 'dec'}"
    log = X.Resumable(tag)
    todo = [r for r in df.index if not log.has(int(r))]
    if not todo:
        X.log(f"{tag}: complete ({len(df)})")
        return
    X.log(f"{tag}: {len(todo)} of {len(df)} to do")
    gp = dict(zip(df.index, gate_probs(df)))
    t0 = time.time()
    for n, r in enumerate(todo, 1):
        row = df.loc[r]
        fb = X.L.build_flow_block(row, X.CFG)
        ids = X.encode(X.L.build_prompt(fb, X.CFG))
        try:
            dec = X.decide(srv, ids, full=full)
        except Exception as e:  # server died: restart once and retry this flow
            X.log(f"  {tag}: server error ({type(e).__name__}); restarting")
            srv.stop()
            srv.start(timeout=900)
            dec = X.decide(srv, ids, full=full)
        rec = {"row": int(r), "true": str(row["Attack_type"]), "label": dec["label"],
               "conf": dec["conf"], "parsed": dec["parsed"], "prefix_found": dec["prefix_ids"] is not None,
               "flow_block": fb, "n_prompt": len(ids), "t_decide": round(dec["seconds"], 3),
               "gate": gp[r]}
        if full:
            rec["text"] = dec["text"]
            try:
                obj = json.loads(dec["text"])
                rec.update(rationale=obj.get("rationale"), action=obj.get("action"),
                           action_rationale=obj.get("action_rationale"))
            except Exception:
                rec.update(rationale=None, action=None, action_rationale=None)
        if trie_on:
            t1 = time.time()
            base = ids + (dec["prefix_ids"] if dec["prefix_ids"] else trie.prefix_ids)
            dist, info = X.class_distribution(srv, trie, base)
            rec.update(dist=dist, missing=info["missing"],
                       root_mass=info["raw_valid_mass"].get("[]"), t_trie=round(time.time() - t1, 3))
        log.write(int(r), rec)
        if n % 100 == 0:
            el = time.time() - t0
            X.log(f"  {tag}: {n}/{len(todo)}  {el / n:.2f} s/flow  ~{(len(todo) - n) * el / n / 60:.0f} min left")
    log.close()


JOBS = {
    "A": [("q8_s456", "val_cal", False, True), ("q8_s456", "test", False, True),
          ("q8_s456", "realistic", False, True)],
    "R": [("q8_s456", "test_sub1500", True, False)],
    "S": [("q8_s42", "val_sub1500", False, True), ("q8_s42", "test_sub1500", False, True),
          ("q8_s123", "val_sub1500", False, True), ("q8_s123", "test_sub1500", False, True)],
    "Q": [("q4_s456", "val_sub1500", False, True), ("q4_s456", "test_sub1500", False, True),
          ("q4_s456", "test_sub1500", True, False)],
    "F": [("bf16_s456", "val_sub300", False, True), ("bf16_s456", "test_sub300", False, True)],
}
NGL = {"bf16_s456": 20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="A,R,S,Q,F")
    a = ap.parse_args()
    draws = build_draws()
    X.log("draws: " + ", ".join(f"{k}={len(v)}" for k, v in draws.items()))
    trie = X.ClassTrie(X.L.build_prompt(X.L.build_flow_block(draws["test"].iloc[0], X.CFG), X.CFG))
    current, srv = None, None
    try:
        for job in a.jobs.split(","):
            for model, draw, full, trie_on in JOBS[job]:
                if model != current:
                    if srv:
                        srv.stop()
                    X.verify_model(model)
                    srv = X.Server(model, port=8091, ngl=NGL.get(model, 99))
                    X.log(f"loading {model} ... {srv.start(timeout=900):.0f}s")
                    current = model
                run(model, draw, draws[draw], full, trie_on, srv, trie)
    finally:
        if srv:
            srv.stop()
    X.log("all requested jobs complete")


if __name__ == "__main__":
    main()
