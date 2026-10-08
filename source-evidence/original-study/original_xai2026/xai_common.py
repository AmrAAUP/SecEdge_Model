"""XAI-2026 shared machinery.

Everything the experiment scripts need: the token-id prompt path (the only path that
reproduces the deployed model), a llama-server process wrapper, the class-trie scorer
that recovers the model's intrinsic 15-class distribution, resumable per-flow logs, and
the calibration / selective-prediction metrics.

Reuses the thesis library (transfer/bundle/secedge_lib.py) for the prompt renderer and
the Stage-1 gate so the prompts are byte-identical to the instruction-tuning corpus.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
FIGS = ROOT / "figs"
OUT.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

BUNDLE = Path("D:/FiveCapterThis/transfer/bundle")
sys.path.insert(0, str(BUNDLE))
import secedge_lib as L  # noqa: E402

CFG = L.load_config()
CLASSES: list[str] = list(CFG["classes"])
SAMPLES = BUNDLE / "artifacts" / "samples"
SPLITS = Path("D:/FiveCapterThis/transfer/data/splits")

SERVER_EXE = Path("D:/FiveCapterThis/pi5_h3_measure/llama/cuda/llama-server.exe")
GRAMMAR_DIR = Path("D:/FiveCapterThis/pi5_h3_measure")
GRAMMAR_FULL = (GRAMMAR_DIR / "secedge.gbnf").read_text(encoding="utf-8")
GRAMMAR_DECISION = (GRAMMAR_DIR / "secedge_decision.gbnf").read_text(encoding="utf-8")

# ONLY checkpoints whose SHA-256 matches the Raspberry Pi's verified copy are used.
# The copies under D:/Lthesis/models and D:/1111111 are corrupted (same size, different
# bytes) and produce garbage; they are deliberately not referenced here.
MODELS = {
    "q8_s456": "D:/مرجع/thises/secedge-3.8b-q8_0-seed456.gguf",
    "q8_s42": "D:/مرجع/thises/secedge-3.8b-q8_0-seed42.gguf",
    "q8_s123": "D:/مرجع/thises/secedge-3.8b-q8_0-seed123.gguf",
    "bf16_s456": "D:/مرجع/thises/secedge-3.8b-bf16-seed456.gguf",
    "q4_s456": "D:/xai_models/q4_s456.gguf",  # pulled from the board
}
EXPECTED_SHA256 = {
    "q8_s456": "c7aaaeefa586f616a1d91fbd007cfbfc7e1844baf7a399e6809cf4d24745d40a",
    "q8_s42": "178f3ea5e20039e9e75c0622d8cd1378c5a0d66dc12b8b6438187dbb34f71d24",
    "q8_s123": "83810c836b7d5058b49c9716e11d6c1e7193d45328109e345dbe167e3dd64153",
    "bf16_s456": "7ce52993c71fab63ccac7cacc6d0f36983a6f436ecf7721eff966eb7c083c221",
    "q4_s456": "43c53b8a116d8f6d96c4ad621adfe3fee835bac24318073e15a25055bef4026e",
}

LABEL_PREFIX = '{"label": "'


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# tokenizer — the HF tokenizer that reproduces the oracle 300/300
# --------------------------------------------------------------------------

_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(str(ROOT / "tokenizer"), local_files_only=True)
    return _TOK


def encode(text: str) -> list[int]:
    return tokenizer()(text, add_special_tokens=False)["input_ids"]


def prompt_for_row(row) -> str:
    return L.build_prompt(L.build_flow_block(row, CFG), CFG)


# --------------------------------------------------------------------------
# class trie — exact constrained-decoding mass over the 15 labels
# --------------------------------------------------------------------------

class ClassTrie:
    """Token paths of every class label as the corpus tokenised them.

    Paths are taken from the joint tokenisation of prompt + target so any
    boundary effect after <|assistant|> is reproduced exactly; construction
    fails loudly if the prompt ids are not a stable prefix.
    """

    def __init__(self, sample_prompt: str):
        pids = encode(sample_prompt)
        paths = {}
        for c in CLASSES:
            full = encode(sample_prompt + LABEL_PREFIX + c + '"')
            if full[:len(pids)] != pids:
                raise RuntimeError(f"prompt ids are not a stable prefix for class {c}")
            paths[c] = full[len(pids):]
        # shared label prefix = longest common prefix of all paths
        lcp = 0
        while all(len(p) > lcp for p in paths.values()) and \
                len({p[lcp] for p in paths.values()}) == 1:
            lcp += 1
        self.prefix_ids = next(iter(paths.values()))[:lcp]
        self.paths = {c: p[lcp:] for c, p in paths.items()}
        # a path must not be a prefix of another, otherwise the closing quote is needed
        for a, pa in self.paths.items():
            for b, pb in self.paths.items():
                if a != b and pb[:len(pa)] == pa:
                    raise RuntimeError(f"class path {a} is a prefix of {b}")
        # branching nodes: prefixes (tuples) that have >= 2 distinct next tokens
        children: dict[tuple, set] = {}
        for p in self.paths.values():
            for k in range(len(p)):
                children.setdefault(tuple(p[:k]), set()).add(p[k])
        self.children = children
        self.branch_nodes = [node for node, ch in children.items() if len(ch) >= 2]

    def describe(self) -> dict:
        tok = tokenizer()
        return {
            "prefix_ids": self.prefix_ids,
            "prefix_text": tok.decode(self.prefix_ids),
            "paths": {c: p for c, p in self.paths.items()},
            "path_tokens": {c: [tok.decode([t]) for t in p] for c, p in self.paths.items()},
            "branch_nodes": [list(n) for n in self.branch_nodes],
        }


# --------------------------------------------------------------------------
# llama-server process wrapper
# --------------------------------------------------------------------------

class Server:
    def __init__(self, model_key: str, port: int = 8091, ngl: int = 99, ctx: int = 2048,
                 threads: int = 4, log_name: str | None = None):
        self.model = MODELS[model_key]
        self.model_key = model_key
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.args = [str(SERVER_EXE), "-m", self.model, "--port", str(port), "-ngl", str(ngl),
                     "-c", str(ctx), "-np", "1", "-t", str(threads)]
        self.log_path = OUT / (log_name or f"server_{model_key}.log")
        self.proc = None

    def start(self, timeout: float = 300.0) -> float:
        t0 = time.time()
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(self.args, stdout=self._log, stderr=subprocess.STDOUT)
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {self.log_path}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if json.loads(r.read().decode()).get("status") == "ok":
                        return time.time() - t0
            except Exception:
                pass
            time.sleep(1.0)
        raise TimeoutError("llama-server did not become healthy")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "_log", None):
            self._log.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def completion(self, ids: list[int], n_predict: int, grammar: str | None = None,
                   n_probs: int = 0, stop: list[str] | None = None, timeout: float = 600,
                   cache_prompt: bool = True) -> dict:
        payload = {"prompt": ids, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
                   "cache_prompt": cache_prompt, "n_probs": n_probs, "stream": False}
        if grammar:
            payload["grammar"] = grammar
        if stop:
            payload["stop"] = stop
        req = urllib.request.Request(self.url + "/completion", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())


def _top_probs(entry: dict) -> dict[int, float]:
    """Token-id -> probability from one completion_probabilities entry (both schemas)."""
    tops = entry.get("top_logprobs") or entry.get("top_probs") or entry.get("probs") or []
    out = {}
    for t in tops:
        tid = t.get("id")
        if tid is None:
            continue
        if "logprob" in t:
            out[int(tid)] = math.exp(t["logprob"])
        elif "prob" in t:
            out[int(tid)] = float(t["prob"])
    return out


def class_distribution(server: Server, trie: ClassTrie, base_ids: list[int],
                       n_probs: int = 500) -> tuple[dict[str, float], dict]:
    """Exact renormalised label distribution by querying only the branching nodes.

    base_ids = prompt ids + the ids that precede the label value (either the prefix the
    deployed grammar decoder actually produced, or the corpus-canonical prefix).
    A class whose first token falls outside the returned top-n_probs gets mass 0 and is
    counted in `missing` so the truncation is visible, never silent.
    """
    node_probs: dict[tuple, dict[int, float]] = {}
    raw_mass, missing = {}, 0
    for node in trie.branch_nodes:
        kids = trie.children[node]
        # adaptive top-k: widen until every valid child is returned, so no class is
        # silently given zero mass (tail probabilities matter for conformal scores)
        for k_probs in (n_probs, 5000, 32064):
            res = server.completion(base_ids + list(node), n_predict=1, n_probs=k_probs)
            cp = res.get("completion_probabilities") or []
            tp = _top_probs(cp[0]) if cp else {}
            if all(k in tp for k in kids):
                break
        masses = {}
        for k in kids:
            if k in tp:
                masses[k] = tp[k]
            else:
                masses[k] = 0.0
                missing += 1
        raw_mass[str(list(node))] = float(sum(masses.values()))
        z = sum(masses.values())
        node_probs[node] = ({k: v / z for k, v in masses.items()} if z > 0
                            else {k: 1.0 / len(kids) for k in kids})
    dist = {}
    for c, path in trie.paths.items():
        p = 1.0
        for k in range(len(path)):
            node = tuple(path[:k])
            if node in node_probs:
                p *= node_probs[node][path[k]]
        dist[c] = p
    z = sum(dist.values())
    dist = {c: v / z for c, v in dist.items()}
    return dist, {"raw_valid_mass": raw_mass, "missing": missing}


_LABEL_OPEN = None


def parse_decision(text: str) -> tuple[str | None, float | None, bool]:
    import re
    t = text.strip()
    try:
        obj = json.loads(t if t.endswith("}") else t + "}")
        return obj.get("label"), float(obj.get("confidence")), True
    except Exception:
        m = re.search(r'"label"\s*:\s*"([^"]+)"', text)
        c = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        return (m.group(1) if m else None, float(c.group(1)) if c else None, False)


def decide(server: Server, prompt_ids: list[int], full: bool = False) -> dict:
    """Greedy grammar-constrained decision exactly as deployed.

    Returns the label, verbalised confidence, the raw text, and the ids of the tokens the
    decoder produced before the label value (used to score the label distribution at the
    very prefix the deployed system saw).
    """
    import re
    global _LABEL_OPEN
    if _LABEL_OPEN is None:
        _LABEL_OPEN = re.compile(r'\s*\{\s?"label":\s?"$')
    grammar = GRAMMAR_FULL if full else GRAMMAR_DECISION
    n_pred = 384 if full else 48
    t0 = time.time()
    # cache_prompt=False: the decision is recomputed from scratch, so its numerics never
    # depend on which flow was processed before (order-independent, reproducible). The
    # trie calls that follow reuse this flow's own KV cache, which is itself deterministic.
    res = server.completion(prompt_ids, n_predict=n_pred, grammar=grammar, n_probs=1,
                            stop=["<|end|>"], cache_prompt=False)
    secs = time.time() - t0
    text = res.get("content", "")
    cp = res.get("completion_probabilities") or []
    acc, prefix_ids = "", None
    for i, e in enumerate(cp):
        acc += e.get("token", "")
        if _LABEL_OPEN.match(acc):
            prefix_ids = [int(x["id"]) for x in cp[:i + 1]]
            break
    label, conf, parsed = parse_decision(text)
    return {"label": label, "conf": conf, "parsed": parsed, "text": text,
            "prefix_ids": prefix_ids, "seconds": secs,
            "n_gen": len(cp), "timings": res.get("timings", {})}


def verify_model(key: str) -> str:
    """SHA-256 of a checkpoint, cached by (path, size, mtime); raises on mismatch."""
    path = Path(MODELS[key])
    st = path.stat()
    cache_p = OUT / "model_hashes.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    ck = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    if ck not in cache:
        log(f"hashing {key} ({st.st_size/1e9:.1f} GB) ...")
        cache[ck] = sha256_file(path)
        cache_p.write_text(json.dumps(cache, indent=1))
    if cache[ck] != EXPECTED_SHA256[key]:
        raise RuntimeError(f"{key}: sha256 {cache[ck]} != verified {EXPECTED_SHA256[key]}")
    return cache[ck]


# --------------------------------------------------------------------------
# vectorised flow-block renderer (for whole splits); validated against L.build_flow_block
# --------------------------------------------------------------------------

def render_blocks(df) -> "pd.Series":
    import pandas as pd
    fb = CFG["flow_block"]
    parts = []
    for label, column, rule in fb["field_order"]:
        if column not in df.columns:
            parts.append(pd.Series("", index=df.index))
            continue
        raw = df[column]
        text = raw.astype(str).str.strip()
        present = raw.notna() & (text != "") & (text.str.lower() != "nan")
        if rule == "nonzero":
            num = pd.to_numeric(text, errors="coerce")
            present &= ~(num.notna() & (num == 0.0))
        parts.append((label + "=" + text).where(present, ""))
    sep = fb["separator"]
    acc = parts[0]
    for p in parts[1:]:
        both = (acc != "") & (p != "")
        acc = acc + both.map({True: sep, False: ""}) + p
    return fb["header"] + "\n" + acc


def validate_renderer(df, n: int = 2000, seed: int = 0) -> tuple[int, int]:
    s = df.sample(min(n, len(df)), random_state=seed)
    fast = render_blocks(s)
    ok = sum(fast.loc[i] == L.build_flow_block(s.loc[i], CFG) for i in s.index)
    return ok, len(s)


# --------------------------------------------------------------------------
# resumable per-flow log
# --------------------------------------------------------------------------

class Resumable:
    def __init__(self, name: str):
        self.path = OUT / f"{name}.jsonl"
        self.done = set()
        if self.path.exists():
            for line in open(self.path, encoding="utf-8"):
                line = line.strip()
                if line:
                    try:
                        self.done.add(json.loads(line)["key"])
                    except Exception:
                        pass
        self.fh = open(self.path, "a", encoding="utf-8")

    def has(self, key) -> bool:
        return key in self.done

    def write(self, key, rec: dict) -> None:
        rec = dict(rec, key=key)
        self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.fh.flush()
        self.done.add(key)

    def close(self):
        self.fh.close()


def read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def ece(conf, correct, n_bins: int = 15, equal_mass: bool = False) -> tuple[float, float, list]:
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    n = len(conf)
    if equal_mass:
        order = np.argsort(conf)
        groups = np.array_split(order, n_bins)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
        groups = [np.where(idx == b)[0] for b in range(n_bins)]
    e = m = 0.0
    rows = []
    for g in groups:
        if len(g) == 0:
            continue
        mc, ac = conf[g].mean(), correct[g].mean()
        e += len(g) / n * abs(ac - mc)
        m = max(m, abs(ac - mc))
        rows.append({"n": int(len(g)), "mean_conf": float(mc), "accuracy": float(ac)})
    return float(e), float(m), rows


def brier_binary(conf, correct) -> float:
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    return float(np.mean((conf - correct) ** 2))


def auroc(score, correct) -> float:
    from sklearn.metrics import roc_auc_score
    correct = np.asarray(correct, int)
    if correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct, score))


def aurc(score, correct) -> tuple[float, float]:
    """Area under the risk-coverage curve and its excess over the oracle (E-AURC)."""
    score = np.asarray(score, float)
    err = 1 - np.asarray(correct, float)
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    a = float(cum.mean())
    r = err.mean()
    # oracle ordering: all correct first
    n = len(err)
    k = int(round((1 - r) * n))
    oracle = np.concatenate([np.zeros(k), np.ones(n - k)])
    ocum = np.cumsum(oracle) / np.arange(1, n + 1)
    return a, a - float(ocum.mean())


def selective_accuracy(score, correct, coverage: float) -> float:
    score = np.asarray(score, float)
    correct = np.asarray(correct, float)
    k = max(1, int(round(coverage * len(score))))
    order = np.argsort(-score, kind="stable")[:k]
    return float(correct[order].mean())


def bootstrap_ci(fn, *arrays, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(fn(*[a[idx] for a in arrays]))
    vals = np.asarray(vals, float)
    return float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0))


def sha256_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
