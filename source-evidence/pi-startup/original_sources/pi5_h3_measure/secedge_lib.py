"""
Edge-SecRAG — shared library for the Iteration 3 (H3) on-device measurement bundle.

Everything the phase scripts need lives here: the exact training-time prompt
reconstruction, the Stage-1 XGBoost gate, the llama.cpp SLM wrapper with GBNF
constraint, the RAG store, the Observe-Think-Act-Reflect agent, the C1/C4.1-C4.4
pipeline configurations, the resource monitor, and the statistics used for the
pre-registered verdicts.

No phase script is allowed to invent a number. Anything that could not be
measured is written out as null with a `_status` field explaining why, so the
consolidation step can distinguish "measured and failed" from "not measured".
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else HERE / "config.json"
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_root"] = str(HERE)
    # SECEDGE_RAG_IN_LABEL_PATH=0|1 selects the architecture per run without
    # editing the frozen config; see the comment on that key in config.json.
    env = os.environ.get("SECEDGE_RAG_IN_LABEL_PATH")
    if env is not None:
        cfg["pipeline"]["rag_in_label_path"] = env.strip().lower() in ("1", "true", "yes")
    for key in ("decision_first", "ground_after_decision"):
        env = os.environ.get("SECEDGE_" + key.upper())
        if env is not None:
            cfg["pipeline"][key] = env.strip().lower() in ("1", "true", "yes")
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the bundle directory."""
    return (HERE / cfg["paths"][key]).resolve()


def out_path(cfg: dict, name: str) -> Path:
    d = (HERE / cfg["paths"]["out_dir"]).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=_json_default)
    print(f"  wrote {path}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# feature engineering — Stage-1 XGBoost B2 baseline
# --------------------------------------------------------------------------

def numeric_feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    """The canonical 37-feature numeric set (thesis Section 4.2.2.1).

    All numeric columns, minus the binary Attack_label (label leakage) and the
    five session-nonce / identifier columns listed in config.json. Sorted so the
    column order is deterministic across machines and pandas versions.
    """
    drop = set(cfg["features"]["drop_leakage"]) | set(cfg["features"]["drop_identifier"])
    drop.add("Attack_type")
    cols = [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(cols)


def to_feature_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df.reindex(columns=cols)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.fillna(0.0).to_numpy(dtype=np.float32)


# --------------------------------------------------------------------------
# prompt reconstruction — recovered from data/secedge_train_v1.jsonl
# --------------------------------------------------------------------------

def build_flow_block(row: pd.Series, cfg: dict, mask_fields: set[str] | None = None,
                     normalise_presence: bool = False) -> str:
    """Rebuild the training-time 'Flow features:' block for one flow.

    Field order and the two emission rules were derived from the shipped
    instruction-tuning corpus and verified against it by 01_preflight.py:

      always    emit whenever the cell is non-empty, even if the value is 0
      nonzero   emit only when the value parses as a non-zero number

    The second rule is exact, not heuristic: across all 52,568 corpus examples a
    'nonzero' field is emitted with a zero value zero times.

    Phase 08 (shortcut ablation) uses the two optional arguments to close the two
    artefact channels measured in Edge-IIoTset, which are independent:

      mask_fields         closes the LEXICAL channel — free text sitting in a
                          numeric slot, e.g. '_googlecast._tcp.local' in
                          tcp_srcport on 31.15 % of MITM rows and 0 % elsewhere.
      normalise_presence  closes the STRUCTURAL channel — because 'nonzero'
                          fields are omitted when zero, *which* fields appear is
                          itself a signal. Two presence signatures are 100 % pure
                          MITM and cover 33.3 % of that class. Emitting every
                          field unconditionally removes it.

    Masking values does not close the structural channel, so a complete test runs
    both.
    """
    fb = cfg["flow_block"]
    sc = cfg["shortcut_ablation"]
    mask_value = sc["mask_value"]
    absent_value = sc.get("absent_value", "0")
    parts = []
    for label, column, rule in fb["field_order"]:
        present = column in row.index
        text = ""
        if present:
            value = row[column]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                present = False
            else:
                text = str(value).strip()
                if text == "" or text.lower() == "nan":
                    present = False
        if present and rule == "nonzero":
            try:
                if float(text) == 0.0:
                    present = False
            except ValueError:
                pass  # non-numeric value in a numeric slot: emit it
        if not present:
            if not normalise_presence:
                continue
            text = absent_value
        if mask_fields and label in mask_fields:
            text = mask_value
        parts.append(f"{label}={text}")
    return fb["header"] + "\n" + fb["separator"].join(parts)


def build_prompt(flow_block: str, cfg: dict, rag_context: str = "",
                 self_critique: bool = False) -> str:
    """Phi-3 chat template, byte-identical to the corpus 'text' field."""
    system = cfg["flow_block"]["system_prompt"]
    user = flow_block
    if rag_context:
        user = ("Relevant threat-intelligence context:\n" + rag_context
                + "\n\n" + flow_block)
    if self_critique:
        user = ("Your previous analysis of this flow was low-confidence. "
                "Re-examine the evidence critically against the retrieved "
                "context before answering.\n\n" + user)
    return (f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\n{user}<|end|>\n"
            f"<|assistant|>\n")


# --------------------------------------------------------------------------
# Stage-1 gate
# --------------------------------------------------------------------------

class ClassOrderedClassifier:
    """Wraps a fitted classifier so predict_proba columns follow cfg['classes'].

    Defined at module level so the pickled gate can be loaded by any phase.
    """

    def __init__(self, model, column_order: list[int]):
        self.model = model
        self.column_order = column_order

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, self.column_order]


class Stage1Gate:
    """XGBoost B2 operating as the Iteration 3 Stage-1 confidence gate."""

    def __init__(self, model, feature_cols: list[str], classes: list[str], source: str):
        self.model = model
        self.feature_cols = feature_cols
        self.classes = classes
        self.source = source

    @classmethod
    def load(cls, cfg: dict) -> "Stage1Gate":
        pkl = resolve(cfg, "gate_pkl")
        if not pkl.exists():
            raise FileNotFoundError(
                f"Stage-1 gate not found at {pkl}. Run 02_prepare.py first "
                "(it trains B2 from data/splits/train.csv at seed 42), or copy "
                "the original results/baselines/B2_xgboost_seed42.pkl there.")
        import pickle
        with open(pkl, "rb") as fh:
            blob = pickle.load(fh)
        return cls(blob["model"], blob["feature_cols"], blob["classes"],
                   blob.get("source", str(pkl)))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(to_feature_matrix(df, self.feature_cols))

    def decide(self, proba_row: np.ndarray, tau1: float,
               attack_tau: float | None = None,
               strong_attack_classes: set[str] | None = None) -> tuple[str, float, bool]:
        """Return (label, confidence, fast_pass).

        Fast pass fires when the gate is confidently benign (Normal at >= tau1),
        or — per Section 4.4.2.4, which reports 91.2% fast-pass against a 73%
        Normal share — when the gate is very confident about an attack class it
        is empirically strong on.
        """
        idx = int(np.argmax(proba_row))
        label = self.classes[idx]
        conf = float(proba_row[idx])
        if label == "Normal":
            return label, conf, conf >= tau1
        if (strong_attack_classes and attack_tau is not None
                and label in strong_attack_classes and conf >= attack_tau):
            return label, conf, True
        return label, conf, False


# --------------------------------------------------------------------------
# SLM wrapper
# --------------------------------------------------------------------------

@dataclass
class SlmOutput:
    raw: str
    label: str | None
    confidence: float | None
    rationale: str
    action: str | None
    action_rationale: str
    parsed: bool
    gen_seconds: float
    margin: float | None = None
    margin_method: str = "not_computed"
    decision_seconds: float | None = None
    grounding_seconds: float | None = None


class SLM:
    """llama.cpp wrapper with the GBNF-constrained SecEdge output schema.

    Prefers llama-cpp-python (needed for the top-2 margin); falls back to the
    llama-cli binary, in which case the margin is unavailable and the reflection
    trigger degrades to confidence-only. Which path was taken is recorded in
    `self.backend` and propagated into every result file.
    """

    def __init__(self, model_path: Path, cfg: dict):
        self.model_path = Path(model_path)
        self.cfg = cfg
        p = cfg["pipeline"]
        self.max_new_tokens = p["max_new_tokens"]
        self.temperature = p["temperature"]
        self.n_ctx = p["n_ctx"]
        self.n_threads = p["n_threads"]
        self.grammar_path = resolve(cfg, "grammar")
        self.backend = "none"
        self._llm = None
        self._grammar = None
        self._load_seconds = 0.0
        self._load()

    def _load(self) -> None:
        t0 = time.perf_counter()
        self._server_url = os.environ.get("SECEDGE_SERVER", "").rstrip("/")
        self._grammar_text = Path(self.grammar_path).read_text(encoding="utf-8")
        self._decision_grammar = None
        self._decision_grammar_path = Path(self.grammar_path).with_name(
            "secedge_decision.gbnf")
        self._decision_grammar_text = (
            self._decision_grammar_path.read_text(encoding="utf-8")
            if self._decision_grammar_path.exists() else None)

        # 1. an already-running llama-server (fastest path, and the only one that
        #    works out of the box on Windows, where llama-cpp-python has no wheel)
        if self._server_url:
            if _server_alive(self._server_url):
                self.backend = f"llama-server:{self._server_url}"
                self._load_seconds = time.perf_counter() - t0
                return
            raise RuntimeError(
                f"SECEDGE_SERVER is set to {self._server_url} but no llama-server "
                "is responding there. Start it first (see README).")

        # 2. llama-cpp-python: in-process, and the only backend that can compute
        #    the exact top-2 margin by teacher forcing
        try:
            from llama_cpp import Llama, LlamaGrammar
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                logits_all=False,
                verbose=False,
            )
            self._grammar = LlamaGrammar.from_file(str(self.grammar_path))
            self.backend = "llama-cpp-python"
        except Exception as exc:  # noqa: BLE001
            log(f"  llama-cpp-python unavailable ({exc}); falling back to llama-cli")
            if _which("llama-cli") is None:
                raise RuntimeError(
                    "No inference backend available. Use one of:\n"
                    "  (a) export SECEDGE_SERVER=http://127.0.0.1:8080 with llama-server running\n"
                    "  (b) pip install llama-cpp-python\n"
                    "  (c) put llama-cli on PATH\n"
                    "See the README section 'Running without llama-cpp-python'."
                ) from exc
            self.backend = "llama-cli"
        self._load_seconds = time.perf_counter() - t0

    @property
    def cold_start_seconds(self) -> float:
        return self._load_seconds

    def generate(self, prompt: str) -> SlmOutput:
        t0 = time.perf_counter()
        if self.backend.startswith("llama-server"):
            text = _server_complete(self._server_url, prompt, self._grammar_text,
                                    self.max_new_tokens, self.temperature)
        elif self.backend == "llama-cpp-python":
            res = self._llm(
                prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                grammar=self._grammar,
                stop=["<|end|>"],
                echo=False,
            )
            text = res["choices"][0]["text"]
        else:
            text = _llama_cli_generate(
                self.model_path, prompt, self.grammar_path,
                self.max_new_tokens, self.temperature, self.n_threads)
        dt = time.perf_counter() - t0
        return _parse_output(text, dt)

    def generate_decision(self, prompt: str) -> SlmOutput:
        """Generate only the decision fields: label and confidence.

        The full schema orders the fields label, confidence, rationale, action,
        action_rationale, so reaching the action means generating the whole
        rationale first — which Section 5.6.1 identifies as the dominant latency
        term. But the action is not information the model supplies: Section 3.6.5
        defines a deterministic per-class policy, and the policy module already
        overrides the emitted value below the 0.50 confidence floor. The action
        is a pure function of (label, confidence), and both come first.

        Stopping after confidence therefore yields the complete decision in
        roughly a dozen tokens. Measured on 120 flows at Q8_0, this reproduces
        the full schema's label on 96.7 % of flows and its post-floor action on
        97.5 %, for macro-F1 0.8669 against 0.8727, at 4.86x lower latency.
        """
        grammar = self._decision_grammar_text
        if grammar is None:
            return self.generate(prompt)
        t0 = time.perf_counter()
        if self.backend.startswith("llama-server"):
            text = _server_complete(self._server_url, prompt, grammar, 48,
                                    self.temperature)
        elif self.backend == "llama-cpp-python":
            from llama_cpp import LlamaGrammar
            if self._decision_grammar is None:
                self._decision_grammar = LlamaGrammar.from_string(grammar)
            res = self._llm(prompt, max_tokens=48, temperature=self.temperature,
                            grammar=self._decision_grammar, stop=["<|end|>"],
                            echo=False)
            text = res["choices"][0]["text"]
        else:
            text = _llama_cli_generate(self.model_path, prompt,
                                       self._decision_grammar_path, 48,
                                       self.temperature, self.n_threads)
        dt = time.perf_counter() - t0
        out = _parse_output(text, dt)
        out.decision_seconds = dt
        return out

    def label_margin(self, prompt: str) -> tuple[float | None, str]:
        """Top-2 margin over the 15 class labels from constrained-decode mass.

        Scores every class name by teacher forcing on the shared prefix
        `{"label": "`, rewinding the KV cache between candidates. Returns
        (margin, method). Method is recorded so the thesis can state exactly how
        the margin was obtained.
        """
        if self.backend.startswith("llama-server"):
            return _server_label_margin(self._server_url, prompt, self.cfg["classes"])
        if self.backend != "llama-cpp-python":
            return None, "unavailable_llama_cli_backend"
        classes = self.cfg["classes"]
        prefix = prompt + '{"label": "'
        try:
            llm = self._llm
            prefix_tokens = llm.tokenize(prefix.encode("utf-8"), add_bos=True,
                                         special=True)
            if len(prefix_tokens) + 16 >= self.n_ctx:
                return None, "prefix_exceeds_context"
            llm.reset()
            llm.eval(prefix_tokens)
            base_n = llm.n_tokens
            scores = {}
            for cls in classes:
                cls_tokens = llm.tokenize(cls.encode("utf-8"), add_bos=False,
                                          special=False)
                total = 0.0
                llm.n_tokens = base_n  # rewind to the shared prefix
                logits = llm.scores[base_n - 1]
                for tok in cls_tokens:
                    logprobs = _log_softmax(np.asarray(logits, dtype=np.float64))
                    total += float(logprobs[tok])
                    llm.eval([tok])
                    logits = llm.scores[llm.n_tokens - 1]
                scores[cls] = total
            ordered = sorted(scores.values(), reverse=True)
            probs = _softmax(np.array(ordered, dtype=np.float64))
            margin = float(probs[0] - probs[1])
            return margin, "constrained_decode_teacher_forcing"
        except Exception as exc:  # noqa: BLE001
            return None, f"failed:{type(exc).__name__}"


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x)
    return x - m - np.log(np.sum(np.exp(x - m)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def _which(name: str) -> str | None:
    from shutil import which
    return which(name)


# --- llama-server HTTP backend --------------------------------------------
# The standard llama.cpp deployment mode, and on Windows the only path that
# needs no compiler: download the prebuilt CUDA binaries and run llama-server.

def _server_post(url: str, path: str, payload: dict, timeout: int = 900) -> dict:
    import json as _json
    import urllib.request
    req = urllib.request.Request(
        url + path, data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read().decode("utf-8"))


def _server_alive(url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url + "/health", timeout=10) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        try:
            with urllib.request.urlopen(url + "/props", timeout=10) as resp:
                return resp.status == 200
        except Exception:  # noqa: BLE001
            return False


def _server_complete(url: str, prompt: str, grammar: str,
                     max_tokens: int, temperature: float) -> str:
    res = _server_post(url, "/completion", {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "grammar": grammar,
        "cache_prompt": True,
        "stop": ["<|end|>"],
    })
    return res.get("content", "")


def _server_label_margin(url: str, prompt: str,
                         classes: list[str]) -> tuple[float | None, str]:
    """Top-2 margin from the token distribution at the label position.

    Approximate: llama-server returns per-token top-k probabilities, so classes
    sharing a first token (the four DDoS_* variants) are pooled. The margin is
    therefore a lower bound on the true top-2 separation, which is the
    conservative direction for a reflection trigger — it fires slightly more
    often than the exact margin would. Recorded as such.
    """
    try:
        res = _server_post(url, "/completion", {
            "prompt": prompt + '{"label": "',
            "n_predict": 1,
            "temperature": 0.0,
            "n_probs": 20,
            "cache_prompt": True,
        }, timeout=300)
        probs = res.get("completion_probabilities") or []
        if not probs:
            return None, "server_returned_no_probabilities"
        top = probs[0].get("probs") or probs[0].get("top_logprobs") or []
        pooled: dict[str, float] = {}
        for entry in top:
            tok = (entry.get("tok_str") or entry.get("token") or "").strip()
            p = entry.get("prob")
            if p is None and "logprob" in entry:
                p = math.exp(entry["logprob"])
            if not tok or p is None:
                continue
            for cls in classes:
                if cls.startswith(tok):
                    pooled[cls] = max(pooled.get(cls, 0.0), float(p))
                    break
        if len(pooled) < 2:
            return None, "fewer_than_two_class_tokens_in_top_k"
        ranked = sorted(pooled.values(), reverse=True)
        return float(ranked[0] - ranked[1]), "server_top_k_pooled_by_first_token"
    except Exception as exc:  # noqa: BLE001
        return None, f"failed:{type(exc).__name__}"


def _llama_cli_generate(model: Path, prompt: str, grammar: Path,
                        max_tokens: int, temperature: float, threads: int) -> str:
    cmd = [
        "llama-cli", "-m", str(model), "-p", prompt,
        "-n", str(max_tokens), "--temp", str(temperature),
        "--grammar-file", str(grammar), "-t", str(threads),
        "--no-display-prompt", "--simple-io", "-no-cnv",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    return res.stdout


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_output(text: str, gen_seconds: float) -> SlmOutput:
    match = _JSON_RE.search(text or "")
    if not match:
        return SlmOutput(raw=text, label=None, confidence=None, rationale="",
                         action=None, action_rationale="", parsed=False,
                         gen_seconds=gen_seconds)
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return SlmOutput(raw=text, label=None, confidence=None, rationale="",
                         action=None, action_rationale="", parsed=False,
                         gen_seconds=gen_seconds)
    conf = obj.get("confidence")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    return SlmOutput(
        raw=text,
        label=obj.get("label"),
        confidence=conf,
        rationale=obj.get("rationale", "") or "",
        action=obj.get("action"),
        action_rationale=obj.get("action_rationale", "") or "",
        parsed=True,
        gen_seconds=gen_seconds,
    )


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------

class KnowledgeBase:
    """On-device MITRE ATT&CK retrieval store.

    Uses whichever embedding and index backend is actually installed and records
    the choice, because a degraded backend changes what the RAG configurations
    (C4.2-C4.4) mean. `self.degraded` is True whenever the embedder is not the
    pre-registered all-MiniLM-L6-v2.
    """

    def __init__(self, docs: list[dict], embeddings: np.ndarray,
                 embedder, backend: str, degraded: bool):
        self.docs = docs
        self.embeddings = embeddings
        self.embedder = embedder
        self.backend = backend
        self.degraded = degraded

    @classmethod
    def load(cls, cfg: dict) -> "KnowledgeBase":
        kb_dir = resolve(cfg, "kb_dir")
        docs_path = kb_dir / "docs.jsonl"
        emb_path = kb_dir / "embeddings.npy"
        meta_path = kb_dir / "meta.json"
        if not (docs_path.exists() and emb_path.exists()):
            raise FileNotFoundError(
                f"Knowledge base not built at {kb_dir}. Run 02_prepare.py "
                "(--build-kb) first.")
        docs = [json.loads(l) for l in open(docs_path, encoding="utf-8")]
        embeddings = np.load(emb_path)
        meta = json.loads(open(meta_path, encoding="utf-8").read())
        embedder, backend, degraded = _make_embedder(meta["embedder"])
        return cls(docs, embeddings, embedder, backend, degraded)

    def retrieve(self, query: str, k: int, token_cap: int) -> tuple[str, list[dict]]:
        q = self.embedder([query])[0]
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = self.embeddings @ q
        order = np.argsort(-sims)[:k]
        picked, block, used = [], [], 0
        for i in order:
            doc = self.docs[int(i)]
            piece = f"[{doc['id']}] {doc['title']}: {doc['text']}"
            approx_tokens = len(piece) // 4
            if used + approx_tokens > token_cap and picked:
                continue  # drop lowest-scoring document that overruns the cap
            picked.append({**doc, "score": float(sims[int(i)])})
            block.append(piece)
            used += approx_tokens
        return "\n".join(block), picked


def _make_embedder(name: str):
    """Return (callable, backend_name, degraded)."""
    if name == "all-MiniLM-L6-v2":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            def embed(texts):
                return np.asarray(model.encode(texts, normalize_embeddings=True),
                                  dtype=np.float32)

            return embed, "sentence-transformers/all-MiniLM-L6-v2", False
        except Exception as exc:  # noqa: BLE001
            log(f"  sentence-transformers unavailable ({exc}); using hashing fallback")
    return _hashing_embedder(), "hashing-tfidf-fallback", True


def _hashing_embedder(dim: int = 384):
    """Deterministic hashing embedder. Degraded: no semantic generalisation."""
    token_re = re.compile(r"[a-z0-9_.]+")

    def embed(texts):
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in token_re.findall(str(t).lower()):
                out[i, hash(tok) % dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / (norms + 1e-12)

    return embed


# --------------------------------------------------------------------------
# agent
# --------------------------------------------------------------------------

@dataclass
class FlowResult:
    true_label: str
    pred_label: str | None
    confidence: float | None
    margin: float | None
    action: str | None
    cycles: int
    reflected: bool
    fast_pass: bool
    stage: str                 # "gate" | "slm"
    latency_s: float
    parsed: bool
    schema_ok: bool
    mitre_ids: list[str] = field(default_factory=list)
    gate_label: str | None = None
    gate_conf: float | None = None
    decision_latency_s: float | None = None
    grounding_latency_s: float | None = None


_MITRE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")


class CyclicAgent:
    """Observe-Think-Act-Reflect loop with the pre-registered trigger and floor."""

    def __init__(self, slm: SLM, kb: KnowledgeBase | None, cfg: dict,
                 use_rag: bool, use_reflection: bool, max_cycles: int | None = None):
        p = cfg["pipeline"]
        self.slm = slm
        self.kb = kb
        self.cfg = cfg
        self.use_rag = use_rag
        self.use_reflection = use_reflection
        # See config.json: selects between the §3.10.2 architecture (retrieval in
        # the label-producing prompt) and the §4.4.2.3 one (label from a bare
        # in-distribution call, retrieval for grounding only).
        self.rag_in_label_path = p.get("rag_in_label_path", True)
        # decision_first: emit label+confidence only on the critical path and
        # take the action from the policy table; ground afterwards, off-path.
        self.decision_first = p.get("decision_first", False)
        self.ground_after_decision = p.get("ground_after_decision", True)
        self.conf_threshold = p["reflection_confidence_threshold"]
        self.margin_threshold = p["reflection_margin_threshold"]
        self.action_floor = p["action_confidence_floor"]
        self.max_cycles = max_cycles if max_cycles is not None else p["agent_max_cycles"]
        self.top_k = p["rag_top_k"]
        self.top_k_reflect = p["rag_top_k_on_reflection"]
        self.token_cap = p["rag_context_token_cap"]
        self.action_set = set(p["action_set"])

    def run(self, flow_block: str, gate_label: str | None) -> tuple[SlmOutput, int, bool, float]:
        """Return (final_output, cycles, reflected, stage2_seconds)."""
        t0 = time.perf_counter()
        context = ""
        if self.use_rag and self.kb is not None:
            query = f"{gate_label or ''} {flow_block}"
            context, _ = self.kb.retrieve(query, self.top_k, self.token_cap)

        if self.decision_first:
            # Critical path: label + confidence only. The action follows from the
            # deterministic policy table of Section 3.6.5, so nothing else is
            # needed to decide. Grounding is generated afterwards, off the path,
            # and its cost is recorded separately rather than hidden.
            out = self.slm.generate_decision(build_prompt(flow_block, self.cfg))
            decision_s = out.decision_seconds
            out.action = _default_action(self.cfg, out.label)
            if self.ground_after_decision:
                t_g = time.perf_counter()
                grounded = self.slm.generate(
                    build_prompt(flow_block, self.cfg, context))
                out.grounding_seconds = time.perf_counter() - t_g
                if grounded.parsed:
                    out.rationale = grounded.rationale
            out.decision_seconds = decision_s
        elif self.rag_in_label_path or not context:
            out = self.slm.generate(build_prompt(flow_block, self.cfg, context))
        else:
            # Label-preserving architecture: the label and confidence come from a
            # bare, in-distribution call — the prompt shape the model was tuned
            # on — and the retrieval-enriched call supplies only the rationale
            # and action grounding. The label from pass 1 is authoritative.
            base = self.slm.generate(build_prompt(flow_block, self.cfg))
            enriched = self.slm.generate(build_prompt(flow_block, self.cfg, context))
            out = self._merge(base, enriched)
        cycles, reflected = 1, False

        if self.use_reflection:
            while cycles < self.max_cycles:
                margin, method = self.slm.label_margin(
                    build_prompt(flow_block, self.cfg, context))
                out.margin, out.margin_method = margin, method
                low_conf = (out.confidence is None
                            or out.confidence < self.conf_threshold)
                low_margin = margin is not None and margin < self.margin_threshold
                if not (low_conf or low_margin):
                    break
                if self.use_rag and self.kb is not None:
                    context, _ = self.kb.retrieve(
                        f"{out.label or gate_label or ''} {flow_block}",
                        self.top_k_reflect, self.token_cap)
                revised = self.slm.generate(
                    build_prompt(flow_block, self.cfg, context, self_critique=True))
                # Under the label-preserving architecture the reflection cycle
                # revises the rationale, action and confidence but cannot move
                # the label — which is what §4.4.2.3 asserts, and which makes the
                # H3.3 macro-F1 lift structurally zero rather than measurable.
                out = revised if self.rag_in_label_path else self._merge(out, revised)
                cycles += 1
                reflected = True

        # bounded-autonomy policy gate: the 0.50 action floor, distinct from the
        # 0.70 reflection trigger above.
        if out.confidence is not None and out.confidence < self.action_floor:
            out.action = "escalate"
        if out.action not in self.action_set:
            out.action = "escalate"

        return out, cycles, reflected, time.perf_counter() - t0

    @staticmethod
    def _merge(authoritative: SlmOutput, enriched: SlmOutput) -> SlmOutput:
        """Keep the label from the authoritative pass, take grounding from the other.

        Confidence follows the label, so it is kept from the authoritative pass
        too: a confidence emitted alongside a different label would not describe
        the label actually returned.
        """
        if not enriched.parsed:
            return authoritative
        return SlmOutput(
            raw=enriched.raw,
            label=authoritative.label,
            confidence=authoritative.confidence,
            rationale=enriched.rationale or authoritative.rationale,
            action=enriched.action or authoritative.action,
            action_rationale=enriched.action_rationale or authoritative.action_rationale,
            parsed=authoritative.parsed,
            gen_seconds=authoritative.gen_seconds + enriched.gen_seconds,
            margin=authoritative.margin,
            margin_method=authoritative.margin_method,
        )


# --------------------------------------------------------------------------
# pipelines
# --------------------------------------------------------------------------

PIPELINES = {
    "C1":   dict(gate=False, rag=False, reflect=False,
                 desc="Bare SecEdge-3.8B, no gate, no RAG, no agent"),
    "C4.1": dict(gate=True,  rag=False, reflect=False,
                 desc="Stage-1 gate + bare Stage-2 SLM"),
    "C4.2": dict(gate=True,  rag=True,  reflect=False,
                 desc="C4.1 + RAG retrieval block (k=3)"),
    "C4.3": dict(gate=True,  rag=True,  reflect=True,
                 desc="C4.2 + single-cycle confidence-driven reflection"),
    "C4.4": dict(gate=True,  rag=True,  reflect=True,
                 desc="Full selective two-stage pipeline, bounded multi-cycle"),
}


def run_pipeline(name: str, df: pd.DataFrame, gate: Stage1Gate | None,
                 slm: SLM, kb: KnowledgeBase | None, cfg: dict,
                 tau1: float | None = None,
                 strong_attack_classes: set[str] | None = None,
                 mask_fields: set[str] | None = None,
                 normalise_presence: bool = False,
                 progress_every: int = 25,
                 checkpoint: Path | None = None) -> list[FlowResult]:
    """Execute one configuration over `df` and return per-flow results.

    With `checkpoint` set, each flow's result is appended to a JSONL file as it
    completes and completed flows are skipped on a re-run. A multi-hour phase can
    then survive an interruption — a timed-out shell, a dropped SSH session, a
    Pi that reboots — and resume where it stopped instead of starting over.
    Flows are processed in `df` order, so resumption is positional and exact.
    """
    done: list[FlowResult] = []
    if checkpoint is not None and Path(checkpoint).exists():
        with open(checkpoint, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.append(FlowResult(**json.loads(line)))
        if done:
            log(f"    resuming {name} from checkpoint: {len(done)} flows already done")
        if len(done) >= len(df):
            return done[:len(df)]
    spec = PIPELINES[name]
    tau1 = tau1 if tau1 is not None else cfg["pipeline"]["tau1_gate_threshold"]
    max_cycles = 2 if name == "C4.3" else cfg["pipeline"]["agent_max_cycles"]
    agent = CyclicAgent(slm, kb, cfg, spec["rag"], spec["reflect"], max_cycles)

    proba = gate.predict_proba(df) if (spec["gate"] and gate is not None) else None
    results: list[FlowResult] = list(done)
    start_at = len(done)
    ck = open(checkpoint, "a", encoding="utf-8") if checkpoint is not None else None

    for pos, (_, row) in enumerate(df.iterrows()):
        if pos < start_at:
            continue
        t0 = time.perf_counter()
        true_label = str(row["Attack_type"])
        gate_label = gate_conf = None
        fast_pass = False

        if proba is not None:
            gate_label, gate_conf, fast_pass = gate.decide(
                proba[pos], tau1, cfg["gate"]["attack_gate_threshold"],
                strong_attack_classes)

        if fast_pass:
            action = "allow" if gate_label == "Normal" else _default_action(cfg, gate_label)
            results.append(FlowResult(
                true_label=true_label, pred_label=gate_label, confidence=gate_conf,
                margin=None, action=action, cycles=0, reflected=False,
                fast_pass=True, stage="gate",
                latency_s=time.perf_counter() - t0, parsed=True, schema_ok=True,
                mitre_ids=[], gate_label=gate_label, gate_conf=gate_conf))
        else:
            flow_block = build_flow_block(row, cfg, mask_fields, normalise_presence)
            out, cycles, reflected, _ = agent.run(flow_block, gate_label)
            results.append(FlowResult(
                true_label=true_label, pred_label=out.label, confidence=out.confidence,
                margin=out.margin, action=out.action, cycles=cycles,
                reflected=reflected, fast_pass=False, stage="slm",
                latency_s=time.perf_counter() - t0, parsed=out.parsed,
                schema_ok=out.parsed and out.label in cfg["classes"],
                mitre_ids=sorted(set(_MITRE_RE.findall(out.rationale))),
                gate_label=gate_label, gate_conf=gate_conf,
                decision_latency_s=out.decision_seconds,
                grounding_latency_s=out.grounding_seconds))

        if ck is not None:
            ck.write(json.dumps(asdict(results[-1]), default=_json_default) + "\n")
            ck.flush()
            os.fsync(ck.fileno())

        if progress_every and (pos + 1) % progress_every == 0:
            log(f"    {name}: {pos + 1}/{len(df)} flows")

    if ck is not None:
        ck.close()
    return results


def strong_attack_classes(gate: Stage1Gate, df: pd.DataFrame, cfg: dict) -> set[str]:
    """Attack classes the Stage-1 gate may fast-pass without SLM review.

    Derived once from the VALIDATION split and cached, not from the evaluation
    draw. Two reasons, and the first is a correctness matter rather than a
    preference:

      - Section 3.10.1 registers the criterion as "measured on the validation
        split". Deriving it from the evaluation draw instead would let the gate
        configuration be chosen with sight of the data it is scored on.
      - The qualifying set is otherwise a property of the draw's class mix. On a
        stratified draw every class supplies plenty of high-confidence
        predictions and 9-11 classes qualify; on a realistic operational mix the
        rarer attack classes supply fewer than the ten the criterion needs and
        only 4 qualify, which collapses the fast-pass rate from the ~91% the
        design depends on to ~75% and with it the whole H3.2 speedup.

    Set SECEDGE_STRONG_FROM_DRAW=1 to restore the old per-draw behaviour for
    comparison; the choice is recorded in the cache file either way.
    """
    cache = resolve(cfg, "gate_pkl").parent / "gate_strong_classes.json"
    if cache.exists() and not os.environ.get("SECEDGE_STRONG_FROM_DRAW"):
        return set(json.loads(cache.read_text(encoding="utf-8"))["classes"])

    if os.environ.get("SECEDGE_STRONG_FROM_DRAW"):
        source, sample = "evaluation draw (legacy behaviour)", df
    else:
        source = "validation split (per Section 3.10.1)"
        val = pd.read_csv(resolve(cfg, "val_csv"), low_memory=False)
        sample = stratified_sample(val, 2000, cfg["classes"],
                                   cfg["sampling"]["sample_seed"])

    proba = gate.predict_proba(sample)
    truth = sample["Attack_type"].to_numpy()
    argmax = np.argmax(proba, axis=1)
    conf = proba[np.arange(len(sample)), argmax]
    pred = np.array([cfg["classes"][i] for i in argmax])
    tau = cfg["gate"]["attack_gate_threshold"]
    strong, detail = set(), {}
    for c in cfg["classes"]:
        if c == "Normal":
            continue
        sel = (pred == c) & (conf >= tau)
        n = int(sel.sum())
        prec = float((truth[sel] == c).mean()) if n else 0.0
        detail[c] = {"n_high_conf": n, "precision": round(prec, 4)}
        if n >= 10 and prec >= 0.99:
            strong.add(c)

    if not os.environ.get("SECEDGE_STRONG_FROM_DRAW"):
        cache.write_text(json.dumps(
            {"classes": sorted(strong), "source": source,
             "n_sample": int(len(sample)), "attack_tau": tau,
             "per_class": detail}, indent=1), encoding="utf-8")
    log(f"    gate-strong set ({len(strong)}) from {source}: {sorted(strong)}")
    return strong


def _default_action(cfg: dict, label: str | None) -> str:
    policy_path = resolve(cfg, "action_policy")
    if not hasattr(_default_action, "_cache"):
        with open(policy_path, encoding="utf-8") as fh:
            _default_action._cache = json.load(fh)["policy"]
    entry = _default_action._cache.get(label or "", {})
    return entry.get("default_action", "escalate")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def macro_f1(y_true: list[str], y_pred: list[str], classes: list[str]) -> float:
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp == 0 and (fp or fn):
            f1s.append(0.0)
            continue
        if tp == 0:
            continue  # class absent from this sample: excluded from the average
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def per_class_report(y_true: list[str], y_pred: list[str],
                     classes: list[str]) -> dict[str, dict]:
    rep = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        support = tp + fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / support if support else 0.0
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        rep[c] = dict(precision=round(prec, 4), recall=round(rec, 4),
                      f1=round(f1, 4), support=support)
    return rep


def bootstrap_ci(y_true: list[str], y_pred: list[str], classes: list[str],
                 n_resamples: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return dict(point=None, mean=None, ci95_lo=None, ci95_hi=None,
                    n_resamples=0, _status="empty_sample")
    point = macro_f1(y_true, y_pred, classes)
    stats = np.empty(n_resamples, dtype=np.float64)
    yt, yp = np.array(y_true), np.array(y_pred)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        stats[i] = macro_f1(list(yt[idx]), list(yp[idx]), classes)
    return dict(point=round(point, 4), mean=round(float(stats.mean()), 4),
                ci95_lo=round(float(np.percentile(stats, 2.5)), 4),
                ci95_hi=round(float(np.percentile(stats, 97.5)), 4),
                n_resamples=n_resamples)


def paired_bootstrap_delta(y_true: list[str], pred_a: list[str], pred_b: list[str],
                           classes: list[str], n_resamples: int,
                           seed: int = 42) -> dict:
    """Bootstrap of macro-F1(B) - macro-F1(A) on paired predictions."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return dict(delta_pp=None, _status="empty_sample")
    yt, pa, pb = np.array(y_true), np.array(pred_a), np.array(pred_b)
    point = macro_f1(y_true, pred_b, classes) - macro_f1(y_true, pred_a, classes)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        deltas[i] = (macro_f1(list(yt[idx]), list(pb[idx]), classes)
                     - macro_f1(list(yt[idx]), list(pa[idx]), classes))
    p_value = float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())))
    return dict(delta_pp=round(point * 100, 4),
                ci95_lo_pp=round(float(np.percentile(deltas, 2.5)) * 100, 4),
                ci95_hi_pp=round(float(np.percentile(deltas, 97.5)) * 100, 4),
                p_value=p_value, n_resamples=n_resamples, n=n)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict:
    if total == 0:
        return dict(rate=None, lo=None, hi=None, n=0, _status="empty_sample")
    p = successes / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return dict(rate=round(p, 4), lo=round(max(0.0, centre - half), 4),
                hi=round(min(1.0, centre + half), 4), successes=successes, n=total)


def holm_bonferroni(tests: list[dict], alpha: float) -> list[dict]:
    """tests: [{'label':..., 'p_value':...}, ...]. Returns them ranked with verdicts."""
    usable = [t for t in tests if t.get("p_value") is not None]
    skipped = [{**t, "adjusted_alpha": None, "reject_h0": None,
                "_status": "no_p_value_not_measured"}
               for t in tests if t.get("p_value") is None]
    ordered = sorted(usable, key=lambda t: t["p_value"])
    m = len(ordered)
    out, still_rejecting = [], True
    for i, t in enumerate(ordered):
        adj = alpha / (m - i)
        reject = still_rejecting and t["p_value"] <= adj
        if not reject:
            still_rejecting = False
        out.append({**t, "rank": i + 1, "adjusted_alpha": round(adj, 6),
                    "reject_h0": reject})
    return out + skipped


def expected_calibration_error(confidences: list[float], correct: list[bool],
                               n_bins: int = 10) -> dict:
    conf = np.asarray([c for c in confidences if c is not None], dtype=np.float64)
    corr = np.asarray([c for c, k in zip(correct, confidences) if k is not None],
                      dtype=np.float64)
    if conf.size == 0:
        return dict(ECE=None, MCE=None, n=0, bins=[], _status="no_confidence_values")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = mce = 0.0
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(sel.sum())
        if n == 0:
            bins.append(dict(lo=round(float(lo), 2), hi=round(float(hi), 2), n=0))
            continue
        acc, avg_conf = float(corr[sel].mean()), float(conf[sel].mean())
        gap = abs(acc - avg_conf)
        ece += (n / conf.size) * gap
        mce = max(mce, gap)
        bins.append(dict(lo=round(float(lo), 2), hi=round(float(hi), 2), n=n,
                         accuracy=round(acc, 4), mean_confidence=round(avg_conf, 4)))
    return dict(ECE=round(ece, 4), MCE=round(mce, 4), n=int(conf.size), bins=bins)


def latency_percentiles(seconds: list[float]) -> dict:
    if not seconds:
        return dict(_status="empty_sample")
    arr = np.asarray(seconds, dtype=np.float64)
    return dict(n=int(arr.size), mean_s=round(float(arr.mean()), 4),
                p50_s=round(float(np.percentile(arr, 50)), 4),
                p95_s=round(float(np.percentile(arr, 95)), 4),
                p99_s=round(float(np.percentile(arr, 99)), 4),
                max_s=round(float(arr.max()), 4))


def summarise(results: list[FlowResult], cfg: dict, label: str) -> dict:
    classes = cfg["classes"]
    y_true = [r.true_label for r in results]
    y_pred = [r.pred_label if r.pred_label else "UNKNOWN" for r in results]
    stage2 = [r for r in results if r.stage == "slm"]
    reflected = [r for r in stage2 if r.reflected]
    attack = [r for r in results if r.true_label != "Normal"]
    return dict(
        config=label,
        rag_architecture=("rag_in_label_path (§3.10.2: retrieval prepended to the "
                          "label-producing prompt)"
                          if cfg["pipeline"].get("rag_in_label_path", True) else
                          "label_preserving (§4.4.2.3: label from a bare call, "
                          "retrieval for grounding only)"),
        n=len(results),
        macro_f1=round(macro_f1(y_true, y_pred, classes), 4),
        accuracy=round(sum(t == p for t, p in zip(y_true, y_pred)) / max(1, len(results)), 4),
        per_class=per_class_report(y_true, y_pred, classes),
        bootstrap=bootstrap_ci(y_true, y_pred, classes,
                               cfg["sampling"]["bootstrap_resamples"]),
        latency=latency_percentiles([r.latency_s for r in results]),
        stage2_latency=latency_percentiles([r.latency_s for r in stage2]),
        decision_latency=latency_percentiles(
            [r.decision_latency_s for r in results if r.decision_latency_s is not None]),
        amortised_decision_latency_s=(
            round(sum(r.decision_latency_s or 0.0 for r in results) / max(1, len(results)), 4)
            if any(r.decision_latency_s is not None for r in results) else None),
        grounding_latency=latency_percentiles(
            [r.grounding_latency_s for r in results if r.grounding_latency_s is not None]),
        fast_pass_rate=round(sum(r.fast_pass for r in results) / max(1, len(results)), 4),
        stage2_rate=round(len(stage2) / max(1, len(results)), 4),
        reflection_rate_of_stage2=wilson_interval(len(reflected), len(stage2)),
        reflection_rate_of_all=wilson_interval(len(reflected), len(results)),
        median_cycles=int(np.median([r.cycles for r in stage2])) if stage2 else 0,
        schema_failures=sum(not r.schema_ok for r in stage2),
        unknown_predictions=sum(1 for p in y_pred if p == "UNKNOWN"),
        action_compliance=round(
            sum(r.action in set(cfg["pipeline"]["action_set"]) for r in results)
            / max(1, len(results)), 4),
        action_floor_violations=sum(
            1 for r in results
            if r.confidence is not None
            and r.confidence < cfg["pipeline"]["action_confidence_floor"]
            and r.action != "escalate"),
        mitre_grounding_rate=round(
            sum(1 for r in attack if r.stage == "slm" and r.mitre_ids)
            / max(1, len([r for r in attack if r.stage == "slm"])), 4),
        calibration=expected_calibration_error(
            [r.confidence for r in results],
            [r.true_label == r.pred_label for r in results]),
    )


def results_to_records(results: list[FlowResult]) -> list[dict]:
    return [asdict(r) for r in results]


# --------------------------------------------------------------------------
# resource monitor
# --------------------------------------------------------------------------


def _inference_process_rss_gb(psutil_mod) -> float:
    """RSS of the inference runtime, in GB.

    Returns the summed RSS of any llama-server / llama-cli process, or 0.0 when
    the model is held in-process (llama-cpp-python), in which case the caller
    falls back to its own RSS. Reported separately from the harness because the
    H2.2 budget is a claim about a deployed gateway, not about the pandas and
    scikit-learn stack this bundle happens to run alongside it.
    """
    if psutil_mod is None:
        return 0.0
    total = 0.0
    try:
        for pr in psutil_mod.process_iter(["name", "memory_info"]):
            name = (pr.info.get("name") or "").lower()
            if "llama-server" in name or "llama-cli" in name:
                mi = pr.info.get("memory_info")
                if mi:
                    total += mi.rss
    except Exception:  # noqa: BLE001
        return 0.0
    return total / (1024 ** 3)


class ResourceMonitor:
    """Samples process RSS at 100 ms and SoC temperature at 1 s, per Section 3.7."""

    def __init__(self, rss_interval: float = 0.1, temp_interval: float = 1.0):
        self.rss_interval = rss_interval
        self.temp_interval = temp_interval
        self.rss_samples: list[float] = []
        self.self_rss_samples: list[float] = []
        self.inference_rss_samples: list[float] = []
        self.system_used_samples: list[float] = []
        self.temp_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._temp_source = "unavailable"

    def _read_temp(self) -> float | None:
        try:
            res = subprocess.run(["vcgencmd", "measure_temp"],
                                 capture_output=True, text=True, timeout=5)
            m = re.search(r"([\d.]+)", res.stdout)
            if m:
                self._temp_source = "vcgencmd"
                return float(m.group(1))
        except Exception:  # noqa: BLE001
            pass
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as fh:
                self._temp_source = "thermal_zone0"
                return float(fh.read().strip()) / 1000.0
        except Exception:  # noqa: BLE001
            return None

    def _loop(self) -> None:
        try:
            import psutil
            proc = psutil.Process(os.getpid())
        except Exception:  # noqa: BLE001
            psutil = None  # type: ignore
            proc = None
        last_temp = 0.0
        while not self._stop.is_set():
            if proc is not None:
                try:
                    own = proc.memory_info().rss / (1024 ** 3)
                    infer = _inference_process_rss_gb(psutil)
                    self.self_rss_samples.append(own)
                    self.inference_rss_samples.append(infer)
                    self.system_used_samples.append(
                        psutil.virtual_memory().used / (1024 ** 3))
                    # The deployment-relevant figure is the inference runtime:
                    # in-process when llama-cpp-python holds the model, the
                    # server's RSS when it runs separately. The analysis harness
                    # (pandas, sklearn, xgboost) is measurement scaffolding and
                    # would not be present in a deployed gateway.
                    self.rss_samples.append(infer if infer > 0 else own)
                except Exception:  # noqa: BLE001
                    pass
            now = time.time()
            if now - last_temp >= self.temp_interval:
                t = self._read_temp()
                if t is not None:
                    self.temp_samples.append(t)
                last_temp = now
            time.sleep(self.rss_interval)

    def __enter__(self) -> "ResourceMonitor":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def report(self) -> dict:
        def peak(v):
            return round(float(np.max(v)), 4) if v else None
        rss = np.asarray(self.rss_samples) if self.rss_samples else None
        temp = np.asarray(self.temp_samples) if self.temp_samples else None
        return dict(
            peak_rss_gb=round(float(rss.max()), 4) if rss is not None else None,
            mean_rss_gb=round(float(rss.mean()), 4) if rss is not None else None,
            rss_samples=int(rss.size) if rss is not None else 0,
            peak_inference_rss_gb=peak(self.inference_rss_samples),
            peak_harness_rss_gb=peak(self.self_rss_samples),
            peak_system_used_gb=peak(self.system_used_samples),
            rss_definition=("inference runtime only (llama-server/llama-cli); "
                            "the harness and system totals are reported alongside "
                            "so H2.2 can state which definition it uses"
                            if any(v > 0 for v in self.inference_rss_samples)
                            else "in-process model plus analysis harness "
                                 "(llama-cpp-python backend): NOT comparable to a "
                                 "deployed-runtime figure, see peak_harness_rss_gb"),
            temp_p95_c=round(float(np.percentile(temp, 95)), 2) if temp is not None else None,
            temp_max_c=round(float(temp.max()), 2) if temp is not None else None,
            temp_mean_c=round(float(temp.mean()), 2) if temp is not None else None,
            temp_samples=int(temp.size) if temp is not None else 0,
            temp_source=self._temp_source,
            _status=("ok" if rss is not None and temp is not None
                     else "partial: psutil or temperature source unavailable"),
        )


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame, per_class: int, classes: list[str],
                      seed: int) -> pd.DataFrame:
    parts = []
    for c in classes:
        sub = df[df["Attack_type"] == c]
        if len(sub) == 0:
            continue
        parts.append(sub.sample(n=min(per_class, len(sub)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def realistic_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw n flows preserving the Edge-IIoTset operational class mix.

    The mix is measured from the test split rather than hardcoded, so the 73%
    Normal share quoted in Section 4.4.2.4 is reproduced from data.
    """
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def discover_seed_models(cfg: dict) -> list[tuple[int, Path]]:
    """Locate the seed checkpoints to evaluate.

    SECEDGE_GGUF_GLOB overrides the configured pattern. Use it to pick the
    precision the question calls for: the deployment artefact is Q8_0, but the
    thesis's server-side results — Table 4.7 per-class, the C4.1-C4.4 ablation
    (Section 4.4.2.2) and the n = 6,184 check (Section 4.2.2.9) — were all
    measured at bf16, so reproducing those means

        export SECEDGE_GGUF_GLOB='secedge-3.8b-bf16-seed*.gguf'
    """
    glob_pat = os.environ.get("SECEDGE_GGUF_GLOB") or cfg["paths"]["gguf_glob"]
    pattern = str(resolve(cfg, "gguf_dir") / glob_pat)
    found = []
    for p in sorted(glob.glob(pattern)):
        m = re.search(r"seed(\d+)", os.path.basename(p))
        if m:
            found.append((int(m.group(1)), Path(p)))
    return found
