"""Track D — on-device confirmation on the Raspberry Pi 5.

export : build a 150-flow bundle (10/class from test_sub300) with the exact HF token ids,
         the class trie and the decision grammar; copy it and the board runner to the Pi and
         launch the run detached.
fetch  : pull the per-flow results back.
analyze: compare with the GPU (same flows, canonical prefix) and apply the FROZEN Track-A
         temperature and Mondrian thresholds fitted on val on the workstation.
"""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

import xai_common as X

PI = "aaup@192.168.1.38"
PI_DIR = "~/Desktop/pi5_h3_measure"
PI_MODEL = "~/Desktop/paper/paper_q1/pi5_q8/pi5_q8/models/secedge-3.8b-q8_0-seed456.gguf"
BUNDLE = X.OUT / "pi_bundle.json"
RESULTS = X.OUT / "pi_results.jsonl"

BOARD_RUNNER = r'''
import json, os, sys, time
import numpy as np
from llama_cpp import Llama, LlamaGrammar
B = json.load(open(os.path.expanduser("~/Desktop/pi5_h3_measure/xai_pi_bundle.json")))
OUT = os.path.expanduser("~/Desktop/pi5_h3_measure/out_pi/xai_pi_results.jsonl")
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["row"])
        except Exception: pass
llm = Llama(model_path=os.path.expanduser(B["model"]), n_ctx=2048, n_threads=4, n_batch=512,
            logits_all=True, verbose=False)   # scores[] is only filled when logits_all=True
gram = LlamaGrammar.from_string(B["grammar"], verbose=False)
pre = B["prefix_ids"]; nodes = B["branch_nodes"]; kids = B["children"]
fh = open(OUT, "a")
def softmax(z):
    z = np.asarray(z, dtype=np.float64); z = z - z.max(); e = np.exp(z); return e / e.sum()
for n, r in enumerate(B["flows"], 1):
    if r["row"] in done: continue
    llm.reset()                      # fresh evaluation: order-independent numerics
    t0 = time.time()
    o = llm.create_completion(prompt=r["ids"], max_tokens=48, temperature=0.0, grammar=gram,
                              stop=["<|end|>"])
    t_dec = time.time() - t0
    text = o["choices"][0]["text"]
    t1 = time.time()
    node_probs = {}
    for node in nodes:
        llm.n_tokens = len(r["ids"])  # rewind to the prompt; KV beyond is discarded by eval
        llm.eval(pre + node)
        p = softmax(llm.scores[llm.n_tokens - 1])
        node_probs[json.dumps(node)] = {str(k): float(p[k]) for k in kids[json.dumps(node)]}
    t_trie = time.time() - t1
    fh.write(json.dumps({"row": r["row"], "text": text, "node_probs": node_probs,
                         "t_decide": round(t_dec, 3), "t_trie": round(t_trie, 3)}) + "\n")
    fh.flush()
    print(f"{n}/{len(B['flows'])} dec {t_dec:.1f}s trie {t_trie:.1f}s  {text[:50]}", flush=True)
print("DONE", flush=True)
'''


def export():
    sub = pd.read_csv(X.OUT / "draws" / "test_sub300_index.csv")
    rows = []
    for c in X.CLASSES:
        rows += sub[sub["Attack_type"] == c]["row"].head(10).tolist()
    test = pd.read_csv(X.SAMPLES / "largesample_n6184.csv", low_memory=False)
    trie = X.ClassTrie(X.prompt_for_row(test.loc[rows[0]]))
    flows = [{"row": int(r), "true": str(test.loc[r, "Attack_type"]),
              "ids": X.encode(X.prompt_for_row(test.loc[r]))} for r in rows]
    bundle = {"model": PI_MODEL, "grammar": X.GRAMMAR_DECISION, "prefix_ids": trie.prefix_ids,
              "branch_nodes": [list(n) for n in trie.branch_nodes],
              "children": {json.dumps(list(n)): sorted(trie.children[n]) for n in trie.branch_nodes},
              "paths": trie.paths, "flows": flows}
    json.dump(bundle, open(BUNDLE, "w"))
    runner = X.OUT / "xai_pi_run.py"
    runner.write_text(BOARD_RUNNER, encoding="utf-8")
    for src, dst in ((BUNDLE, "xai_pi_bundle.json"), (runner, "xai_pi_run.py")):
        subprocess.run(["scp", "-o", "BatchMode=yes", str(src), f"{PI}:Desktop/pi5_h3_measure/{dst}"], check=True)
    cmd = (f"cd {PI_DIR} && source .venv/bin/activate && mkdir -p out_pi && "
           f"nohup python3 -u xai_pi_run.py > out_pi/xai_pi_run.log 2>&1 & echo launched")
    print(subprocess.run(["ssh", "-o", "BatchMode=yes", PI, cmd], capture_output=True, text=True).stdout)
    print(f"exported {len(flows)} flows")


def fetch():
    subprocess.run(["scp", "-o", "BatchMode=yes", f"{PI}:Desktop/pi5_h3_measure/out_pi/xai_pi_results.jsonl",
                    str(RESULTS)], check=True)
    print(sum(1 for _ in open(RESULTS)), "results")


def dist_from_nodes(node_probs: dict, trie_paths: dict, children: dict) -> dict:
    norm = {}
    for node, probs in node_probs.items():
        z = sum(probs.values())
        norm[node] = {int(k): v / z for k, v in probs.items()}
    out = {}
    for c, path in trie_paths.items():
        p = 1.0
        for k in range(len(path)):
            key = json.dumps(path[:k])
            if key in norm:
                p *= norm[key][path[k]]
        out[c] = p
    z = sum(out.values())
    return {c: v / z for c, v in out.items()}


def analyze():
    import x2_trackA as A
    B = json.load(open(BUNDLE))
    pi = {r["row"]: r for r in X.read_jsonl(RESULTS)}
    gpu_ref = {r["row"]: r for r in X.read_jsonl(X.OUT / "x6_gpu_reference.jsonl")}
    ta = json.load(open(X.OUT / "x2_trackA_q8_s456_val_cal.json"))
    T = ta["temperature"]
    q = np.array(ta["tests"]["test"]["conformal"]["lac_mondrian_0.05"]["q"])
    rows = [f["row"] for f in B["flows"] if f["row"] in pi and f["row"] in gpu_ref]
    truth = {f["row"]: f["true"] for f in B["flows"]}
    agree, dpm, cov, size, tdec, ttrie, lab_pi, lab_gpu = [], [], [], [], [], [], [], []
    for r in rows:
        dp = dist_from_nodes(pi[r]["node_probs"], B["paths"], B["children"])
        dg = gpu_ref[r]["dist_canonical"]
        lp, _, _ = X.parse_decision(pi[r]["text"])
        lab_pi.append(lp)
        lab_gpu.append(gpu_ref[r]["label"])
        agree.append(lp == gpu_ref[r]["label"])
        dpm.append(abs(max(dp.values()) - max(dg.values())))
        P = np.array([[dp[c] for c in X.CLASSES]])
        Pt = A.apply_T(P, T)[0]
        s = (1 - Pt) <= q
        y = X.CLASSES.index(truth[r])
        cov.append(bool(s[y]))
        size.append(int(s.sum()))
        tdec.append(pi[r]["t_decide"])
        ttrie.append(pi[r]["t_trie"])
    n = len(rows)
    lo, hi = _wilson(sum(cov), n)
    res = {"n": n, "label_agreement_gpu_pi": float(np.mean(agree)),
           "mean_abs_delta_pmax": float(np.mean(dpm)), "max_abs_delta_pmax": float(np.max(dpm)),
           "pi_accuracy": float(np.mean([a == truth[r] for a, r in zip(lab_pi, rows)])),
           "gpu_accuracy": float(np.mean([a == truth[r] for a, r in zip(lab_gpu, rows)])),
           "pi_conformal_coverage_0.05": float(np.mean(cov)), "coverage_wilson95": [lo, hi],
           "pi_mean_set_size": float(np.mean(size)), "pi_singleton_rate": float(np.mean(np.array(size) == 1)),
           "pi_decide_s_mean": float(np.mean(tdec)), "pi_trie_s_mean": float(np.mean(ttrie)),
           "trie_overhead_ratio": float(np.mean(ttrie) / np.mean(tdec))}
    res["D-T1"] = res["label_agreement_gpu_pi"] >= 0.99 and res["mean_abs_delta_pmax"] <= 0.02
    res["D-T2"] = res["trie_overhead_ratio"] <= 0.10
    json.dump(res, open(X.OUT / "x6_trackD.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "fetch", "analyze"])
    {"export": export, "fetch": fetch, "analyze": analyze}[ap.parse_args().cmd]()
