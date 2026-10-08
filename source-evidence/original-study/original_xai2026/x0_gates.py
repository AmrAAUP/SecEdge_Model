"""Phase 0 gates — nothing downstream runs unless every gate passes.

G0  the checkpoint is byte-identical to the Raspberry Pi's verified copy
G1  the HF tokenizer reproduces the oracle token ids 300/300
G2  grammar-constrained greedy decisions on the 300-flow confirmatory draw reproduce the
    published token-id result (workstation 0.9321 / board 0.9356 macro-F1, +-0.005),
    with every output parseable
G3  the class-trie distribution agrees with the greedy label on >= 99% of flows and no
    class first-token falls outside the returned top-k
G4  re-querying 50 flows returns identical labels and confidences
"""
import json
import time

import numpy as np
import pandas as pd

import xai_common as X

ORACLE = "D:/Lthesis/tokenid/prompts/tokenid_prompts.json"
MODEL = "q8_s456"


def main():
    rep = {"model": MODEL, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rep["G0_sha256"] = X.verify_model(MODEL)
    X.log(f"G0 ok: {rep['G0_sha256'][:16]}...")

    tp = json.load(open(ORACLE, encoding="utf-8"))
    ok = sum(X.encode(r["prompt"]) == r["token_ids"] for r in tp["records"])
    rep["G1_tokenizer"] = {"match": ok, "n": len(tp["records"]), "pass": ok == len(tp["records"])}
    X.log(f"G1 tokenizer oracle {ok}/{len(tp['records'])}")

    d = pd.read_csv(X.SAMPLES / "confirmatory_n300.csv", low_memory=False)
    truth = [str(x) for x in d["Attack_type"]]
    prompts = [X.prompt_for_row(d.iloc[i]) for i in range(len(d))]
    trie = X.ClassTrie(prompts[0])
    json.dump(trie.describe(), open(X.OUT / "trie.json", "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)

    srv = X.Server(MODEL, port=8091)
    rep["server_load_s"] = round(srv.start(), 1)
    run = X.Resumable(f"x0_confirmatory_{MODEL}")
    try:
        for i, p in enumerate(prompts):
            if run.has(i):
                continue
            ids = X.encode(p)
            dec = X.decide(srv, ids)
            t0 = time.time()
            base = ids + (dec["prefix_ids"] if dec["prefix_ids"] else trie.prefix_ids)
            dist, info = X.class_distribution(srv, trie, base)
            t_trie = time.time() - t0
            dist_c, info_c = X.class_distribution(srv, trie, ids + trie.prefix_ids)
            run.write(i, {"i": i, "true": truth[i], "label": dec["label"], "conf": dec["conf"],
                          "parsed": dec["parsed"], "text": dec["text"],
                          "prefix_ids": dec["prefix_ids"], "dist": dist, "dist_canonical": dist_c,
                          "raw_mass": info["raw_valid_mass"], "missing": info["missing"],
                          "missing_canonical": info_c["missing"],
                          "t_decide": round(dec["seconds"], 3), "t_trie": round(t_trie, 3)})
            if (i + 1) % 25 == 0:
                X.log(f"  {i + 1}/300")
        run.close()

        recs = sorted(X.read_jsonl(run.path), key=lambda r: r["i"])
        # G4 determinism: re-query the first 50 decisions + distributions
        diffs, same = [], 0
        for r in recs[:50]:
            ids = X.encode(prompts[r["i"]])
            dec = X.decide(srv, ids)
            base = ids + (dec["prefix_ids"] if dec["prefix_ids"] else trie.prefix_ids)
            dist, _ = X.class_distribution(srv, trie, base)
            same += int(dec["label"] == r["label"] and dec["conf"] == r["conf"])
            diffs.append(max(abs(dist[c] - r["dist"][c]) for c in X.CLASSES))
    finally:
        srv.stop()

    labels = [r["label"] for r in recs]
    f1 = X.macro_f1(truth, labels)
    acc = float(np.mean([a == b for a, b in zip(truth, labels)]))
    parsed = sum(r["parsed"] for r in recs)
    argmax = [max(r["dist"], key=r["dist"].get) for r in recs]
    argmax_c = [max(r["dist_canonical"], key=r["dist_canonical"].get) for r in recs]
    agree = float(np.mean([a == b for a, b in zip(argmax, labels)]))
    agree_c = float(np.mean([a == b for a, b in zip(argmax_c, labels)]))
    pmax_gap = float(np.mean([abs(max(r["dist"].values()) - max(r["dist_canonical"].values()))
                              for r in recs]))
    missing = sum(r["missing"] for r in recs)
    root_mass = [r["raw_mass"].get("[]") for r in recs]

    rep["G2_reproduction"] = {
        "macro_f1": round(f1, 4), "accuracy": round(acc, 4), "parsed": parsed, "n": len(recs),
        "reference_workstation": 0.9321, "reference_board": 0.9356,
        "pass": parsed == len(recs) and (abs(f1 - 0.9321) <= 0.005 or abs(f1 - 0.9356) <= 0.005)}
    rep["G3_trie"] = {
        "argmax_agrees_with_greedy": round(agree, 4),
        "argmax_agrees_canonical_prefix": round(agree_c, 4),
        "mean_abs_pmax_gap_deployed_vs_canonical": round(pmax_gap, 4),
        "missing_first_tokens": missing,
        "root_valid_mass_mean": round(float(np.mean(root_mass)), 4),
        "root_valid_mass_min": round(float(np.min(root_mass)), 4),
        "prefix_found": sum(1 for r in recs if r["prefix_ids"]),
        "pass": agree >= 0.99 and missing == 0}
    rep["G4_determinism"] = {"identical_label_conf": same, "n": 50,
                             "max_abs_dp": round(float(max(diffs)), 6), "pass": same == 50}
    rep["timing_s_per_flow"] = {
        "decide_mean": round(float(np.mean([r["t_decide"] for r in recs])), 3),
        "trie_mean": round(float(np.mean([r["t_trie"] for r in recs])), 3)}
    rep["ALL_PASS"] = all(rep[k]["pass"] for k in
                          ("G1_tokenizer", "G2_reproduction", "G3_trie", "G4_determinism"))
    json.dump(rep, open(X.OUT / "x0_gates.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
