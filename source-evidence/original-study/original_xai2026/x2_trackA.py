"""Track A — calibrated confidence, conformal prediction sets, ambiguity localization.

Everything is fitted on the validation calibration draw and scored ONCE on the frozen test
draws. The primary calibrator and primary ranking signal are chosen by 5-fold CV on the
validation draw only (never on test). Refuses to run without the pre-registration file.

usage: python x2_trackA.py --model q8_s456 --val val_cal --tests test,realistic
"""
import argparse
import json
import math
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import xai_common as X

PREREG = X.ROOT / "PREREG_XAI.json"
ALPHAS = (0.10, 0.05, 0.01)
K = len(X.CLASSES)
CI = {c: i for i, c in enumerate(X.CLASSES)}
EPS = 1e-12


# ------------------------------------------------------------------ features
def load(model: str, draw: str) -> list[dict]:
    return X.read_jsonl(X.OUT / f"x1_{model}_{draw}_dec.jsonl")


def feats(recs: list[dict]) -> dict:
    P = np.array([[r["dist"][c] for c in X.CLASSES] for r in recs], float)
    lab = np.array([CI.get(r["label"], -1) for r in recs])
    y = np.array([CI[r["true"]] for r in recs])
    v = np.array([r["conf"] if r["conf"] is not None else 0.0 for r in recs], float)
    G = np.array([[r["gate"].get(c, 0.0) for c in X.CLASSES] for r in recs], float)
    srt = np.sort(P, 1)
    ent = -(np.clip(P, EPS, 1) * np.log(np.clip(P, EPS, 1))).sum(1)
    return {"P": P, "lab": lab, "y": y, "v": v, "G": G,
            "correct": (lab == y).astype(int),
            "pmax": P[np.arange(len(P)), np.clip(lab, 0, K - 1)],
            "margin": srt[:, -1] - srt[:, -2], "entropy": ent,
            "gmax": G.max(1), "g_of_lab": G[np.arange(len(G)), np.clip(lab, 0, K - 1)],
            "agree": (G.argmax(1) == lab).astype(int),
            "blocks": [r["flow_block"] for r in recs]}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack_X(f):
    return np.column_stack([_logit(f["v"]), np.log(np.clip(f["pmax"], EPS, 1)), f["margin"],
                            f["entropy"], _logit(f["gmax"]), _logit(f["g_of_lab"]), f["agree"]])


# ------------------------------------------------------------------ calibrators
def fit_T(P, y):
    L = np.log(np.clip(P, EPS, 1))

    def nll(t):
        Z = L / t
        Z = Z - Z.max(1, keepdims=True)
        lp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()
    return float(minimize_scalar(nll, bounds=(0.05, 50), method="bounded").x)


def apply_T(P, T):
    Z = np.log(np.clip(P, EPS, 1)) / T
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


class Calib:
    """fit on a feature dict, predict P(correct) for another."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, f):
        k = self.kind
        if k == "verbal_isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(f["v"], f["correct"])
        elif k == "verbal_hist":
            q = np.quantile(f["v"], np.linspace(0, 1, 16))
            self.edges = np.unique(q)
            idx = np.clip(np.searchsorted(self.edges, f["v"], side="right") - 1, 0, len(self.edges) - 2)
            self.acc = np.array([f["correct"][idx == b].mean() if (idx == b).any() else f["correct"].mean()
                                 for b in range(len(self.edges) - 1)])
        elif k == "intrinsic_ts":
            self.T = fit_T(f["P"], f["y"])
        elif k == "stack":
            Xs = stack_X(f)
            self.mu, self.sd = Xs.mean(0), Xs.std(0) + 1e-9
            self.m = LogisticRegression(C=1.0, max_iter=2000).fit((Xs - self.mu) / self.sd, f["correct"])
        return self

    def predict(self, f):
        k = self.kind
        if k == "verbal_raw":
            return f["v"]
        if k == "intrinsic_raw":
            return f["pmax"]
        if k == "verbal_isotonic":
            return self.m.predict(f["v"])
        if k == "verbal_hist":
            idx = np.clip(np.searchsorted(self.edges, f["v"], side="right") - 1, 0, len(self.edges) - 2)
            return self.acc[idx]
        if k == "intrinsic_ts":
            Pt = apply_T(f["P"], self.T)
            return Pt[np.arange(len(Pt)), np.clip(f["lab"], 0, K - 1)]
        if k == "stack":
            return self.m.predict_proba((stack_X(f) - self.mu) / self.sd)[:, 1]


KINDS = ["verbal_raw", "verbal_hist", "verbal_isotonic", "intrinsic_raw", "intrinsic_ts", "stack"]


def subset(f, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx]) for k, v in f.items()}


def cv_select(fv):
    """5-fold CV on validation: mean ECE and AUROC per calibrator."""
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    res = {k: {"ece": [], "auroc": [], "aurc": []} for k in KINDS}
    for tr, te in skf.split(np.zeros(len(fv["y"])), fv["correct"]):
        a, b = subset(fv, tr), subset(fv, te)
        for k in KINDS:
            s = Calib(k).fit(a).predict(b)
            res[k]["ece"].append(X.ece(s, b["correct"])[0])
            res[k]["auroc"].append(X.auroc(s, b["correct"]))
            res[k]["aurc"].append(X.aurc(s, b["correct"])[0])
    summ = {k: {m: float(np.nanmean(v)) for m, v in d.items()} for k, d in res.items()}
    primary_cal = min(KINDS, key=lambda k: summ[k]["ece"])
    primary_rank = max(KINDS, key=lambda k: summ[k]["auroc"])
    return summ, primary_cal, primary_rank


# ------------------------------------------------------------------ metrics
def metric_block(s, c, boot=2000):
    s, c = np.asarray(s, float), np.asarray(c, int)
    e, m, _ = X.ece(s, c)
    out = {"ece": e, "ece_mass": X.ece(s, c, equal_mass=True)[0], "mce": m,
           "brier": X.brier_binary(s, c),
           "nll": float(-np.mean(c * np.log(np.clip(s, 1e-6, 1)) + (1 - c) * np.log(np.clip(1 - s, 1e-6, 1)))),
           "auroc": X.auroc(s, c)}
    out["aurc"], out["eaurc"] = X.aurc(s, c)
    for cov in (0.8, 0.9, 0.95):
        out[f"selacc_{int(cov * 100)}"] = X.selective_accuracy(s, c, cov)
    if boot:
        out["ece_ci"] = X.bootstrap_ci(lambda a, b: X.ece(a, b)[0], s, c, n_boot=boot)
        out["auroc_ci"] = X.bootstrap_ci(X.auroc, s, c, n_boot=boot)
        out["aurc_ci"] = X.bootstrap_ci(lambda a, b: X.aurc(a, b)[0], s, c, n_boot=boot)
    return out


def paired_delta(s1, s2, c, fn, n_boot=10000, seed=1):
    rng = np.random.default_rng(seed)
    s1, s2, c = map(np.asarray, (s1, s2, c))
    base = fn(s1, c) - fn(s2, c)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(c), len(c))
        d.append(fn(s1[i], c[i]) - fn(s2[i], c[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(base), "ci": [float(lo), float(hi)]}


# ------------------------------------------------------------------ conformal
def scores(Pt, kind):
    if kind == "lac":
        return 1 - Pt
    order = np.argsort(-Pt, 1)
    S = np.empty_like(Pt)
    cum = np.cumsum(np.take_along_axis(Pt, order, 1), 1)
    np.put_along_axis(S, order, cum, 1)
    return S


def qhat(sc, alpha):
    n = len(sc)
    if n == 0:
        return np.inf
    k = math.ceil((n + 1) * (1 - alpha))
    return np.inf if k > n else float(np.sort(sc)[k - 1])


def conformal(Pv, yv, Pt, yt, kind, alpha, mondrian):
    Sv, St = scores(Pv, kind), scores(Pt, kind)
    true_sv = Sv[np.arange(len(yv)), yv]
    if mondrian:
        q = np.array([qhat(true_sv[yv == c], alpha) for c in range(K)])
    else:
        q = np.full(K, qhat(true_sv, alpha))
    sets = St <= q[None, :]
    size = sets.sum(1)
    cov = sets[np.arange(len(yt)), yt]
    per_class = {}
    for c in range(K):
        m = yt == c
        if m.any():
            n_cal = int((yv == c).sum())
            tol = 1.645 * math.sqrt(alpha * (1 - alpha) * (1 / max(n_cal, 1) + 1 / m.sum()))
            per_class[X.CLASSES[c]] = {"n": int(m.sum()), "coverage": float(cov[m].mean()),
                                       "mean_size": float(size[m].mean()), "tolerance": tol,
                                       "ok": bool(cov[m].mean() >= 1 - alpha - tol)}
    single = size == 1
    return {"coverage": float(cov.mean()), "mean_size": float(size.mean()),
            "singleton_rate": float(single.mean()), "empty_rate": float((size == 0).mean()),
            "acc_on_singletons": float(cov[single].mean()) if single.any() else None,
            "per_class": per_class, "all_classes_ok": all(v["ok"] for v in per_class.values()),
            "q": [float(x) for x in q]}, sets


# ------------------------------------------------------------------ ambiguity
def ambiguous_blocks():
    cache = X.OUT / "train_ambiguous_blocks.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    cols = [c for _, c, _ in X.CFG["flow_block"]["field_order"]] + ["Attack_type"]
    X.log("reading train split for the ambiguity map ...")
    tr = pd.read_csv(X.SPLITS / "train.csv", usecols=lambda c: c in cols, low_memory=False)
    ok, n = X.validate_renderer(tr)
    if ok != n:
        raise RuntimeError(f"vectorised renderer disagrees with build_flow_block on {n - ok}/{n}")
    X.log(f"renderer validated {ok}/{n}; rendering {len(tr)} train blocks ...")
    blocks = X.render_blocks(tr)
    g = pd.DataFrame({"b": blocks, "y": tr["Attack_type"].astype(str)}).groupby("b")["y"].agg(lambda s: sorted(set(s)))
    amb = {b: labs for b, labs in g.items() if len(labs) >= 2}
    meta = {"train_rows": int(len(tr)), "distinct_blocks": int(len(g)), "ambiguous_blocks": len(amb),
            "renderer_validation": [ok, n]}
    pickle.dump((amb, meta), open(cache, "wb"))
    return amb, meta


def ambiguity_analysis(ft, sets):
    amb, meta = ambiguous_blocks()
    is_amb = np.array([b in amb for b in ft["blocks"]])
    size = sets.sum(1)
    multi = size >= 2
    contains_all = []
    for i in np.where(is_amb)[0]:
        labs = [CI[l] for l in amb[ft["blocks"][i]] if l in CI]
        contains_all.append(all(sets[i, l] for l in labs))
    r_amb = float(multi[is_amb].mean()) if is_amb.any() else float("nan")
    r_non = float(multi[~is_amb].mean())
    boot = X.bootstrap_ci(lambda m, a: (m[a].mean() / max(m[~a].mean(), 1e-9)) if a.any() else np.nan,
                          multi.astype(float), is_amb, n_boot=2000)
    return {"train_meta": meta, "n_test_ambiguous": int(is_amb.sum()),
            "share_test_ambiguous": float(is_amb.mean()),
            "multi_rate_ambiguous": r_amb, "multi_rate_unambiguous": r_non,
            "ratio": r_amb / max(r_non, 1e-9), "ratio_ci": boot,
            "sets_contain_all_colliding_labels": float(np.mean(contains_all)) if contains_all else None,
            "accuracy_ambiguous": float(ft["correct"][is_amb].mean()) if is_amb.any() else None,
            "accuracy_unambiguous": float(ft["correct"][~is_amb].mean()),
            "auroc_setsize_detects_ambiguity": X.auroc(size, is_amb.astype(int)),
            "auroc_entropy_detects_ambiguity": X.auroc(ft["entropy"], is_amb.astype(int))}


# ------------------------------------------------------------------ deferral efficiency
def deferral(score_v, cv, score_t, ct, catch=0.406):
    err_v = score_v[cv == 0]
    thr = float(np.quantile(err_v, catch))  # defer below thr catches ~40.6% of val errors
    d = score_t < thr
    caught = ((ct == 0) & d).sum()
    wasted = ((ct == 1) & d).sum()
    return {"threshold_from_val": thr, "test_catch_rate": float(caught / max((ct == 0).sum(), 1)),
            "test_deferral_rate": float(d.mean()),
            "correct_deferred_per_error_caught": float(wasted / max(caught, 1))}


def baseline_floor(v, c, floor=0.5):
    d = v < floor
    caught = ((c == 0) & d).sum()
    return {"floor": floor, "catch_rate": float(caught / max((c == 0).sum(), 1)),
            "deferral_rate": float(d.mean()),
            "correct_deferred_per_error_caught": float(((c == 1) & d).sum() / max(caught, 1))}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q8_s456")
    ap.add_argument("--val", default="val_cal")
    ap.add_argument("--tests", default="test,realistic")
    ap.add_argument("--ambiguity", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    if not PREREG.exists():
        raise SystemExit("PREREG_XAI.json missing — pre-register before scoring any test draw.")
    prereg_sha = X.sha256_file(PREREG)

    fv = feats(load(a.model, a.val))
    cv_summ, primary_cal, primary_rank = cv_select(fv)
    X.log(f"CV on {a.val}: primary calibrator={primary_cal}, primary ranking={primary_rank}")
    fitted = {k: Calib(k).fit(fv) for k in KINDS}
    T = fitted["intrinsic_ts"].T
    Pv = apply_T(fv["P"], T)

    out = {"model": a.model, "val": a.val, "n_val": int(len(fv["y"])), "prereg_sha256": prereg_sha,
           "val_accuracy": float(fv["correct"].mean()), "cv": cv_summ,
           "primary_calibrator": primary_cal, "primary_ranking": primary_rank,
           "temperature": T, "tests": {}}
    for tname in a.tests.split(","):
        ft = feats(load(a.model, tname))
        res = {"n": int(len(ft["y"])), "accuracy": float(ft["correct"].mean()),
               "macro_f1": X.macro_f1([X.CLASSES[i] for i in ft["y"]],
                                      [X.CLASSES[i] if i >= 0 else "UNK" for i in ft["lab"]]),
               "methods": {}}
        S = {k: np.clip(fitted[k].predict(ft), 0, 1) for k in KINDS}
        for k in KINDS:
            res["methods"][k] = metric_block(S[k], ft["correct"], boot=a.boot)
        c = ft["correct"]
        res["paired_vs_verbal"] = {
            "ece_primary_minus_verbal": paired_delta(S[primary_cal], S["verbal_raw"], c,
                                                     lambda s, cc: X.ece(s, cc)[0]),
            "auroc_primary_minus_verbal": paired_delta(S[primary_rank], S["verbal_raw"], c, X.auroc),
            "aurc_primary_minus_verbal": paired_delta(S[primary_rank], S["verbal_raw"], c,
                                                      lambda s, cc: X.aurc(s, cc)[0])}
        Pt = apply_T(ft["P"], T)
        conf = {}
        sets_primary = None
        for kind in ("lac", "aps"):
            for mond in (True, False):
                for al in ALPHAS:
                    r, sets = conformal(Pv, fv["y"], Pt, ft["y"], kind, al, mond)
                    conf[f"{kind}_{'mondrian' if mond else 'marginal'}_{al}"] = r
                    if kind == "lac" and mond and al == 0.05:
                        sets_primary = sets
        res["conformal"] = conf
        if a.ambiguity:
            res["ambiguity"] = ambiguity_analysis(ft, sets_primary)
        res["deferral"] = {
            "baseline_verbal_floor_0.50": baseline_floor(ft["v"], c),
            "primary_ranking_at_val_catch_0.406": deferral(
                np.clip(fitted[primary_rank].predict(fv), 0, 1), fv["correct"], S[primary_rank], c),
            "verbal_at_val_catch_0.406": deferral(fv["v"], fv["correct"], ft["v"], c)}
        # per-flow export for figures / verification
        np.savez_compressed(X.OUT / f"x2_{a.model}_{tname}_scores.npz",
                            y=ft["y"], lab=ft["lab"], correct=c, Pt=Pt, sets=sets_primary,
                            **{f"s_{k}": S[k] for k in KINDS})
        out["tests"][tname] = res
        X.log(f"{tname}: acc={res['accuracy']:.4f} ECE {primary_cal}={res['methods'][primary_cal]['ece']:.4f} "
              f"(verbal {res['methods']['verbal_raw']['ece']:.4f}); AUROC {primary_rank}="
              f"{res['methods'][primary_rank]['auroc']:.4f} (verbal {res['methods']['verbal_raw']['auroc']:.4f})")
    json.dump(out, open(X.OUT / f"x2_trackA_{a.model}_{a.val}.json", "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
