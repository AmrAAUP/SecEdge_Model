"""Assemble RESULTS.json, the figures and RESULTS.md (with the pre-registered verdict table)."""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import xai_common as X

J = lambda name: json.load(open(X.OUT / name, encoding="utf-8"))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, ACC, WARN, OK = "#1d2b2a", "#8a9896", "#00897a", "#b5540c", "#1f7a4d"


def rel_bins(s, c, n=15):
    edges = np.linspace(0, 1, n + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, n - 1)
    out = []
    for b in range(n):
        m = idx == b
        if m.any():
            out.append((s[m].mean(), c[m].mean(), m.sum()))
    return np.array(out)


def fig_reliability(npz, a, path):
    s_v, s_p, c = npz["s_verbal_raw"], npz[f"s_{a['primary_calibrator']}"], npz["correct"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for ax, s, title in ((axs[0], s_v, "Verbalized confidence (as deployed)"),
                         (axs[1], s_p, f"Calibrated: {a['primary_calibrator']}")):
        b = rel_bins(s, c)
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
        ax.scatter(b[:, 0], b[:, 1], s=10 + 250 * np.sqrt(b[:, 2] / b[:, 2].max()),
                   facecolor="#c9d6d4", edgecolor=INK, zorder=3)
        e = X.ece(s, c)[0]
        ax.set_title(f"{title}\nECE = {e:.4f}", fontsize=8)
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("empirical accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_riskcov(npz, a, path):
    c = npz["correct"]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for key, lab, sty in (("verbal_raw", "verbalized", dict(color=WARN, ls="-")),
                          ("intrinsic_raw", "intrinsic p(label)", dict(color=MUTED, ls="--")),
                          (a["primary_ranking"], f"primary: {a['primary_ranking']}", dict(color=ACC, ls="-", lw=1.8))):
        s = npz[f"s_{key}"]
        order = np.argsort(-s, kind="stable")
        acc = np.cumsum(c[order]) / np.arange(1, len(c) + 1)
        cov = np.arange(1, len(c) + 1) / len(c)
        ax.plot(cov, acc, label=f"{lab} (AURC {X.aurc(s, c)[0]:.4f})", **sty)
    ax.axhline(c.mean(), color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("coverage (fraction auto-handled)")
    ax.set_ylabel("accuracy on retained flows")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(max(0.85, c.mean() - 0.03), 1.001)
    ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_conformal(conf, path):
    pc = conf["per_class"]
    names = list(pc)
    cov = [pc[n]["coverage"] for n in names]
    tol = [pc[n]["tolerance"] for n in names]
    size = [pc[n]["mean_size"] for n in names]
    fig, axs = plt.subplots(2, 1, figsize=(6.6, 4.2), sharex=True)
    x = np.arange(len(names))
    axs[0].bar(x, cov, color=[OK if pc[n]["ok"] else WARN for n in names], width=0.7)
    axs[0].axhline(0.95, color=INK, lw=1, ls="--")
    axs[0].errorbar(x, [0.95] * len(x), yerr=[tol, [0] * len(x)], fmt="none", ecolor=MUTED, capsize=2)
    axs[0].set_ylim(0.8, 1.01)
    axs[0].set_ylabel("coverage (α=0.05)")
    axs[1].bar(x, size, color=ACC, width=0.7)
    axs[1].set_ylabel("mean set size")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_ambiguity(amb, path):
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    vals = [amb["multi_rate_unambiguous"], amb["multi_rate_ambiguous"]]
    ax.bar([0, 1], vals, color=[MUTED, WARN], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["unambiguous input", "ambiguous input\n(block seen with ≥2 labels)"], fontsize=7)
    ax.set_ylabel("share of flows given a multi-label set")
    ax.set_title(f"ratio {amb['ratio']:.1f}× (95% CI {amb['ratio_ci'][0]:.1f}–{amb['ratio_ci'][1]:.1f})", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_faith(b, path):
    keys = [("cf", "model counterfactual"), ("shap", "gate TreeSHAP"), ("rat", "rationale-cited (+fill)"),
            ("rand", "random fields")]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    x = np.arange(len(keys))
    comp = [b["comprehensiveness"][f"{k}@3"] for k, _ in keys]
    suff = [b["sufficiency"][f"{k}@3"] for k, _ in keys]
    for ax, d, t in ((axs[0], comp, "comprehensiveness@3 (higher = evidence matters)"),
                     (axs[1], suff, "sufficiency@3 (lower = evidence suffices)")):
        m = [v["mean"] for v in d]
        err = [[v["mean"] - v["ci95"][0] for v in d], [v["ci95"][1] - v["mean"] for v in d]]
        ax.bar(x, m, yerr=err, color=[ACC, MUTED, WARN, "#c9d6d4"], capsize=3, width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in keys], rotation=30, ha="right", fontsize=7)
        ax.set_title(t, fontsize=8)
        ax.set_ylabel("Δ log p(decision)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_audit(audit, path):
    s8 = audit["summary"]["per_category"]
    s4 = audit.get("compare", {}).get("summary", {}).get("per_category")
    cats = list(s8)
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.bar(x - 0.18, [s8[c]["rate"] for c in cats], width=0.36, color=ACC, label="Q8_0 (deployed)")
    if s4:
        ax.bar(x + 0.18, [s4[c]["rate"] for c in cats], width=0.36, color=WARN, label="Q4_K_M")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in cats], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("share of rationales")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_pi(path):
    import x6_pi as P
    B = json.load(open(P.BUNDLE))
    pi = {r["row"]: r for r in X.read_jsonl(P.RESULTS)}
    ref = {r["row"]: r for r in X.read_jsonl(X.OUT / "x6_gpu_reference.jsonl")}
    xs, ys = [], []
    for f in B["flows"]:
        r = f["row"]
        if r in pi and r in ref:
            dp = P.dist_from_nodes(pi[r]["node_probs"], B["paths"], B["children"])
            xs.append(max(ref[r]["dist_canonical"].values()))
            ys.append(max(dp.values()))
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    ax.scatter(xs, ys, s=10, color=ACC)
    ax.set_xlabel("p(decision) on GPU workstation")
    ax.set_ylabel("p(decision) on Raspberry Pi 5")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def verdicts(R):
    a = R["trackA"]["tests"]["test"]
    pc, pr = R["trackA"]["primary_calibrator"], R["trackA"]["primary_ranking"]
    m = a["methods"]
    c = a["conformal"]["lac_mondrian_0.05"]
    amb = a.get("ambiguity", {})
    d = a["deferral"]["primary_ranking_at_val_catch_0.406"]
    v = []
    v.append(("A-T1", "primary calibrator ECE ≤ 0.03, CI upper ≤ 0.04",
              f"{pc}: ECE {m[pc]['ece']:.4f} [{m[pc]['ece_ci'][0]:.4f}, {m[pc]['ece_ci'][1]:.4f}]",
              m[pc]["ece"] <= 0.03 and m[pc]["ece_ci"][1] <= 0.04))
    v.append(("A-T2", "AUROC(correct vs incorrect) ≥ 0.90", f"{pr}: {m[pr]['auroc']:.4f}", m[pr]["auroc"] >= 0.90))
    v.append(("A-T3", "AURC ≤ 0.0145", f"{pr}: {m[pr]['aurc']:.4f}", m[pr]["aurc"] <= 0.0145))
    v.append(("A-T4", "Mondrian α=0.05: all classes covered; size ≤ 1.30; singletons ≥ 85%",
              f"coverage {c['coverage']:.4f}, all classes ok={c['all_classes_ok']}, size {c['mean_size']:.3f}, "
              f"singletons {c['singleton_rate']:.3f}",
              c["all_classes_ok"] and c["mean_size"] <= 1.30 and c["singleton_rate"] >= 0.85))
    if amb:
        v.append(("A-T5", "multi-label rate ambiguous ≥ 3× unambiguous",
                  f"{amb['multi_rate_ambiguous']:.3f} vs {amb['multi_rate_unambiguous']:.3f} → {amb['ratio']:.1f}×",
                  amb["ratio"] >= 3))
    v.append(("A-T6", "≤ 2 correct deferrals per error caught (val-fixed 40.6% catch)",
              f"{d['correct_deferred_per_error_caught']:.2f} (test catch {d['test_catch_rate']:.3f})",
              d["correct_deferred_per_error_caught"] <= 2))
    au = R["audit"]["summary"]
    mr = R["audit_manual"]
    v.append(("B-T1", "descriptive: hallucinated / unobservable claim rates",
              f"hallucinated values {au['hallucinated_value']['rate']:.3f}; unsupported claims {mr['reference_rate']:.3f} "
              f"[{mr['wilson95'][0]:.3f}, {mr['wilson95'][1]:.3f}] (exhaustive coding); automatic lower bound "
              f"{au['unobservable_claim']['rate']:.3f} (precision {mr['auditor_precision']:.2f}, recall {mr['auditor_recall']:.2f})",
              None))
    b = R["trackB"]
    v.append(("B-T2", "counterfactual comp@3 ≥ 2× random", f"ratio {b['B-T2']['ratio']:.2f}", b["B-T2"]["pass"]))
    v.append(("B-T3", "evidence-bound comp@3 ≥ 1.5× rationale-cited; 0 hallucinated; 0 unobservable",
              f"ratio {b['B-T3']['ratio']:.2f}, hallucinated {b['B-T3']['hallucinated']}, "
              f"unobservable {b['B-T3']['unobservable']}", b["B-T3"]["pass"]))
    v.append(("B-T4", "SLM-vs-SHAP top-3 overlap ≥ 0.6", f"{b['B-T4']['top3_overlap_with_fill']:.3f}",
              b["B-T4"]["pass"]))
    ct = R["trackC"]["C-T1"]
    v.append(("C-T1", "Q4 larger sets and more unobservable claims than Q8 (paired)",
              f"size Δ {ct['paired_diff_q4_minus_q8']:+.3f} [{ct['diff_ci95'][0]:+.3f}, {ct['diff_ci95'][1]:+.3f}]; "
              f"unobservable {ct['unobservable_rate_q8']:.3f}→{ct['unobservable_rate_q4']:.3f}", ct["pass"]))
    dd = R["trackD"]
    v.append(("D-T1", "GPU↔Pi label agreement ≥ 99%, mean |Δp| ≤ 0.02",
              f"agreement {dd['label_agreement_gpu_pi']:.3f}, mean |Δp| {dd['mean_abs_delta_pmax']:.4f}", dd["D-T1"]))
    v.append(("D-T2", "trie overhead ≤ 10% of decision latency", f"{dd['trie_overhead_ratio']:.3f}", dd["D-T2"]))
    return v


def main():
    R = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "prereg_sha256": X.sha256_file(X.ROOT / "PREREG_XAI.json"),
         "gates": J("x0_gates.json"), "trackA": J("x2_trackA_q8_s456_val_cal.json"),
         "audit": J("x3_audit_q8_s456.json"), "audit_manual": J("x3_manual_reference.json"),
         "exploratory_stack_ablation": J("x2_exploratory_stack_ablation.json"), "trackB": J("x4_trackB.json"),
         "trackC": J("x5_trackC.json"), "trackD": J("x6_trackD.json")}
    R["verdicts"] = [{"id": i, "target": t, "measured": m, "pass": p} for i, t, m, p in verdicts(R)]
    json.dump(R, open(X.ROOT / "RESULTS.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    npz = np.load(X.OUT / "x2_q8_s456_test_scores.npz")
    fig_reliability(npz, R["trackA"], X.FIGS / "fig1_reliability.png")
    fig_riskcov(npz, R["trackA"], X.FIGS / "fig2_risk_coverage.png")
    fig_conformal(R["trackA"]["tests"]["test"]["conformal"]["lac_mondrian_0.05"], X.FIGS / "fig3_conformal_per_class.png")
    if "ambiguity" in R["trackA"]["tests"]["test"]:
        fig_ambiguity(R["trackA"]["tests"]["test"]["ambiguity"], X.FIGS / "fig4_ambiguity.png")
    fig_faith(R["trackB"], X.FIGS / "fig5_faithfulness.png")
    fig_audit(R["audit"], X.FIGS / "fig6_claim_audit.png")
    fig_pi(X.FIGS / "fig7_gpu_vs_pi.png")
    X.log("RESULTS.json and figures written")


if __name__ == "__main__":
    main()
