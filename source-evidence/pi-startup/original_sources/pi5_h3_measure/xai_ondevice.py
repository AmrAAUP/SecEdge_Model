"""Fresh on-device XAI confirmation for the ICSIC paper.

Runs the deployed Q8_0 SecEdge-3.8B on a FRESH stratified draw (not the cached
confirmatory_n300) and logs, per flow: true label, predicted label, emitted
confidence, bounded action, rationale text and the MITRE technique IDs it cites.
From this we recompute, on the board, grounding / schema+action compliance /
calibration (ECE) / selective prediction (AURC). Non-destructive: writes only to
out_pi/xai_ondevice.jsonl and is resumable at flow granularity.

Usage:
    python3 xai_ondevice.py [N_PER_CLASS] [smoke]
"""
import os, sys, json, time, re
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get("SECEDGE_ROOT", Path.home() / "Desktop" / "pi5_h3_measure"))
sys.path.insert(0, str(ROOT))
import secedge_lib as L  # noqa: E402

GGUF = os.environ.get("SECEDGE_GGUF_Q8")
if not (GGUF and Path(GGUF).exists()):
    sys.exit(f"model missing: {GGUF!r} (set SECEDGE_GGUF_Q8)")

N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SMOKE = "smoke" in sys.argv[2:]
LARGE = ROOT / "artifacts" / "samples" / "largesample_n6184.csv"
OUT = ROOT / "out_pi" / ("xai_smoke.jsonl" if SMOKE else "xai_ondevice.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

cfg = L.load_config()
df = pd.read_csv(LARGE, low_memory=False)
label_col = "Attack_type" if "Attack_type" in df.columns else (
    "Attack_label" if "Attack_label" in df.columns else None)
if label_col is None:
    sys.exit(f"no label column found; columns tail: {list(df.columns)[-6:]}")
print(f"label column: {label_col}; classes: {df[label_col].nunique()}", flush=True)

# fresh stratified draw (concat per class to avoid the groupby().apply() column drop)
parts = []
for cls, g in df.groupby(label_col):
    parts.append(g.sample(min(N_PER, len(g)), random_state=2026))
draw = pd.concat(parts).sample(frac=1.0, random_state=2026).reset_index(drop=True)
if SMOKE:
    draw = draw.head(2).reset_index(drop=True)
truth = [str(x) for x in draw[label_col]]
blocks = [L.build_flow_block(draw.iloc[i], cfg) for i in range(len(draw))]
print(f"draw: {len(draw)} flows, {draw[label_col].nunique()} classes", flush=True)

done = set()
if OUT.exists():
    for line in open(OUT, encoding="utf-8"):
        try:
            done.add(json.loads(line)["i"])
        except Exception:
            pass
fh = open(OUT, "a", encoding="utf-8")
MITRE = re.compile(r"T\d{4}(?:\.\d{3})?")
FLOOR = 0.50
ALL = {"allow", "block_ip", "isolate_device", "drop_packet", "escalate"}

slm = L.SLM(GGUF, cfg)
print("backend:", getattr(slm, "backend", "?"), flush=True)
t0 = time.time()
n_done = 0
for i, block in enumerate(blocks):
    if i in done:
        continue
    t = time.time()
    out = slm.generate(L.build_prompt(block, cfg))
    conf = out.confidence if out.confidence is not None else 0.0
    act = out.action if out.action in ALL else "escalate"
    if conf < FLOOR:
        act = "escalate"
    rec = dict(i=i, true=truth[i], pred=out.label or "UNK", conf=conf, action=act,
               parsed=bool(out.parsed), schema_ok=bool(out.parsed),
               rationale=(out.rationale or "")[:400],
               mitre_ids=sorted(set(MITRE.findall(out.rationale or ""))),
               seconds=round(time.time() - t, 3))
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush(); os.fsync(fh.fileno())
    n_done += 1
    if n_done <= 2 or (i + 1) % 5 == 0:
        el = time.time() - t0
        print(f"{i+1}/{len(draw)}  {el/60:.1f} min  {el/max(n_done,1):.1f} s/flow  "
              f"[{truth[i]} -> {rec['pred']} conf={conf:.2f} act={act} mitre={rec['mitre_ids']}]",
              flush=True)
fh.close()
print(f"DONE {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
