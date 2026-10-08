"""Track B1 — claim-level audit of full-schema rationales against the exact model input.

For every rationale:
  hallucinated values   IPs / ports / numbers that do not occur in the flow block the model read
                        (an absent 'nonzero' field legitimately means 0)
  unobservable claims   quantities or patterns the single-flow input cannot contain: packet counts,
                        byte sizes, durations, rates/timing, payload entropy, multi-flow sequences
  technique correctness cited ATT&CK id vs the TRUE class's technique (parent/child ids match)
  template fidelity     similarity to the nearest shipped template (values masked)

usage: python x3_audit.py --model q8_s456 [--compare q4_s456]
"""
import argparse
import difflib
import json
import math
import re

import numpy as np

import xai_common as X

TEMPL = json.load(open(X.BUNDLE / "templates" / "rationale_templates.json", encoding="utf-8"))
MITRE = {c: v["mitre_id"] for c, v in TEMPL.items() if c != "_meta"}
NONZERO = {lab for lab, _, rule in X.CFG["flow_block"]["field_order"] if rule == "nonzero"}
ALL_FIELDS = [lab for lab, _, _ in X.CFG["flow_block"]["field_order"]]

RE_TID = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
RE_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
RE_PORT = re.compile(r"(?:\bports?\s+(?:on\s+)?|:)(\d{1,5}(?:\.\d+)?)\b", re.I)
RE_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.]*\d)")
# plural "requests" only: "HTTP 0 request to 0" is a filled field value, not a request count
UNIT_AFTER = re.compile(r"^\s*(?:\)|B\b|KB\b|MB\b|bytes?\b|packets?\b|pkts?\b|connection attempts|requests\b|"
                        r"s\b|sec\w*|ms\b|seconds?\b|pps\b|per second|attempts?\b|flows?\b)", re.I)
UNIT_BEFORE = re.compile(r"(?:packets?|pkts|payload|bytes?|duration|rate|in|over|within|size)\s*\(?\s*$", re.I)
NEUTRAL_NUM = re.compile(r"(?:Layer[- ]?\d|Top ?10|\b(?:19|20)\d{2}\b|ATT&CK|IPv4|IPv6|Q8|Q4|\d+-way)", re.I)

UNOBS = {
    "packet_count": r"\bpackets?\b|\bpkts\b|connection attempts|\bpacket counts?\b",
    "byte_size": r"\bbytes?\b|\bpayload (?:size|length)\b|\(\s*(?:\d+|unknown)\s*B\s*\)|\bbulk[- ]write\b",
    "timing_rate": r"\b(?:rate|cadence|inter-?arrival|interval|periodic\w*|beacon\w*|burst\w*|duration|"
                   r"per second|pps|timing|tempo)\b|\bin \d+(?:\.\d+)?\s?s\b",
    "entropy_content": r"\bentropy\b|\bhigh-entropy\b|\bencrypted payload\b",
    # patterns asserted as OBSERVED across several flows; class names such as "flood" or
    # "distributed" restate the label and are deliberately not counted
    "multi_flow": r"\bsequential\b|\bsequence of\b|\bmultiple (?:destination )?ports\b|\brange of (?:destination )?ports\b|"
                  r"\bpreceded by\b|\bfollowed by\b|\bover time\b|\brepeated\b|\bsource-spread\b|\bmany sources\b",
}
UNOBS_RE = {k: re.compile(v, re.I) for k, v in UNOBS.items()}


def parse_block(fb: str) -> dict:
    body = fb.split("\n", 1)[1] if "\n" in fb else ""
    out = {}
    for part in body.split(X.CFG["flow_block"]["separator"]):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _num(s):
    try:
        return float(s)
    except Exception:
        return None


def allowed_values(block: dict) -> tuple[set, set, set]:
    ips = {v for k, v in block.items() if k in ("src_host", "dst_host")}
    ports = {_num(block[k]) for k in ("tcp_srcport", "tcp_dstport") if k in block and _num(block[k]) is not None}
    if "tcp_dstport" not in block:
        ports.add(0.0)  # nonzero rule: absent means zero
    nums = set()
    for k, v in block.items():
        n = _num(v)
        if n is not None:
            nums.add(n)
        for tok in re.findall(r"\d+(?:\.\d+)?", v):
            nums.add(float(tok))
    for k in NONZERO - set(block):
        nums.add(0.0)
    return ips, ports, nums


def audit_text(text: str, block: dict) -> dict:
    text = text or ""
    ips, ports, nums = allowed_values(block)
    # text reproducing an input field value verbatim is observable by definition: mask it
    # before the numeric checks (longest values first; skip trivially short ones like "0")
    verbatim = text
    for v in sorted({v for v in block.values() if len(v) >= 4}, key=len, reverse=True):
        verbatim = verbatim.replace(v, " VAL ")
    masked = RE_TID.sub(" ", verbatim)
    halluc, unobs_nums, grounded = [], [], []
    for ip in RE_IP.findall(masked):
        (grounded if ip in ips else halluc).append(("ip", ip))
    masked2 = RE_IP.sub(" IPADDR ", masked)
    for m in RE_PORT.finditer(masked2):
        p = _num(m.group(1))
        (grounded if p in ports else halluc).append(("port", m.group(1)))
    masked3 = RE_PORT.sub(" PORT ", masked2)
    masked3 = NEUTRAL_NUM.sub(" ", masked3)
    for m in RE_NUM.finditer(masked3):
        val = _num(m.group(1))
        after, before = masked3[m.end():m.end() + 25], masked3[max(0, m.start() - 25):m.start()]
        if UNIT_AFTER.match(after) or UNIT_BEFORE.search(before):
            unobs_nums.append(m.group(1))
        elif val in nums:
            grounded.append(("num", m.group(1)))
        else:
            halluc.append(("num", m.group(1)))
    cats = {k: bool(r.search(verbatim)) for k, r in UNOBS_RE.items()}
    cats["numeric_unobservable"] = bool(unobs_nums)
    return {"hallucinated": halluc, "grounded": grounded, "unobservable_numbers": unobs_nums,
            "unobservable_categories": [k for k, v in cats.items() if v],
            "has_hallucination": bool(halluc), "has_unobservable": any(cats.values())}


def technique_ok(text: str, true: str) -> tuple[bool, list]:
    cited = RE_TID.findall(text or "")
    tid = MITRE.get(true)
    if tid is None:
        return (len(cited) == 0), cited
    root = tid.split(".")[0]
    return any(c == tid or c.split(".")[0] == root for c in cited), cited


def _mask(s: str) -> str:
    s = RE_TID.sub("TID", s or "")
    s = RE_IP.sub("IP", s)
    s = re.sub(r"\{\w+\}", "X", s)
    return re.sub(r"\d+(?:\.\d+)?", "X", s).lower()


TEMPLATES = [_mask(s) for c, v in TEMPL.items() if c != "_meta" for s in v["rationale_templates"]]


def fidelity(text: str) -> float:
    m = _mask(text)
    return max(difflib.SequenceMatcher(None, m, t).ratio() for t in TEMPLATES)


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def rate(flags):
    k, n = int(sum(flags)), len(flags)
    return {"rate": k / n if n else float("nan"), "k": k, "n": n, "wilson95": wilson(k, n)}


def audit_model(model: str) -> tuple[dict, list]:
    recs = X.read_jsonl(X.OUT / f"x1_{model}_test_sub1500_full.jsonl")
    rows = []
    for r in recs:
        block = parse_block(r["flow_block"])
        a = audit_text(r.get("rationale"), block)
        ok, cited = technique_ok(r.get("rationale"), r["true"])
        rows.append({"row": r["row"], "true": r["true"], "label": r["label"],
                     "correct": r["label"] == r["true"], "rationale": r.get("rationale"),
                     "parsed": r.get("rationale") is not None, **a,
                     "technique_ok": ok, "cited": cited, "fidelity": fidelity(r.get("rationale"))})
    parsed = [x for x in rows if x["parsed"]]
    summ = {"n": len(rows), "parsed": len(parsed),
            "hallucinated_value": rate([x["has_hallucination"] for x in parsed]),
            "unobservable_claim": rate([x["has_unobservable"] for x in parsed]),
            "either": rate([x["has_hallucination"] or x["has_unobservable"] for x in parsed]),
            "per_category": {k: rate([k in x["unobservable_categories"] for x in parsed])
                             for k in list(UNOBS) + ["numeric_unobservable"]},
            "technique_ok_when_label_correct": rate([x["technique_ok"] for x in parsed if x["correct"]]),
            "technique_ok_when_label_wrong": rate([x["technique_ok"] for x in parsed if not x["correct"]]),
            "fidelity_mean": float(np.mean([x["fidelity"] for x in parsed])),
            "fidelity_share_above_0.9": float(np.mean([x["fidelity"] >= 0.9 for x in parsed])),
            "distinct_masked_rationales": len({_mask(x["rationale"]) for x in parsed}),
            "by_correctness": {
                "correct": {"unobservable": rate([x["has_unobservable"] for x in parsed if x["correct"]]),
                            "hallucinated": rate([x["has_hallucination"] for x in parsed if x["correct"]])},
                "wrong": {"unobservable": rate([x["has_unobservable"] for x in parsed if not x["correct"]]),
                          "hallucinated": rate([x["has_hallucination"] for x in parsed if not x["correct"]])}},
            "per_class_unobservable": {c: rate([x["has_unobservable"] for x in parsed if x["true"] == c])
                                       for c in X.CLASSES}}
    return summ, rows


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar on paired binary outcomes."""
    from scipy.stats import binomtest
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    p = binomtest(n01, n, 0.5).pvalue if n else 1.0
    return {"only_second": n01, "only_first": n10, "p_exact": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q8_s456")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    summ, rows = audit_model(a.model)
    out = {"model": a.model, "summary": summ}
    with open(X.OUT / f"x3_audit_{a.model}_rows.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.compare:
        s2, rows2 = audit_model(a.compare)
        out["compare"] = {"model": a.compare, "summary": s2}
        m1 = {r["row"]: r for r in rows if r["parsed"]}
        m2 = {r["row"]: r for r in rows2 if r["parsed"]}
        common = sorted(set(m1) & set(m2))
        out["paired"] = {
            "n": len(common),
            "unobservable": mcnemar([m1[i]["has_unobservable"] for i in common],
                                    [m2[i]["has_unobservable"] for i in common]),
            "hallucinated": mcnemar([m1[i]["has_hallucination"] for i in common],
                                    [m2[i]["has_hallucination"] for i in common])}
        with open(X.OUT / f"x3_audit_{a.compare}_rows.jsonl", "w", encoding="utf-8") as fh:
            for r in rows2:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(out, open(X.OUT / f"x3_audit_{a.model}.json", "w"), indent=1)
    s = summ
    X.log(f"{a.model}: hallucinated {s['hallucinated_value']['rate']:.3f}  unobservable "
          f"{s['unobservable_claim']['rate']:.3f}  technique ok (label right/wrong) "
          f"{s['technique_ok_when_label_correct']['rate']:.3f}/{s['technique_ok_when_label_wrong']['rate']:.3f}")


if __name__ == "__main__":
    main()
