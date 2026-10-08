# XAI-2026 — Results

**SecEdge-3.8B (Phi-3-mini fine-tune, deployed Q8_0) · pre-registered 2026-09-11 00:55 (PREREG_XAI.json, SHA-256 `75bba6db…`) · all numbers re-derived by `verify_xai.py`: 21/21 OK.**

Every figure below comes from per-flow logs in `out/`. Calibrators and conformal thresholds were fitted on the validation split only; the frozen test draw (`largesample_n6184`) was scored once, after pre-registration. Failures are reported as failures.

## Verdicts

| ID | Target | Result | Verdict |
|---|---|---|---|
| A-T1 | Calibrated ECE ≤ 0.03 (CI upper ≤ 0.04) | **0.0094** [0.0052, 0.0159] (deployed verbal: 0.1174) | PASS |
| A-T2 | Correct-vs-incorrect AUROC ≥ 0.90 | **0.9868** (deployed: 0.7322; Δ +0.255 [0.232, 0.278]) | PASS |
| A-T3 | AURC ≤ 0.0145 | **0.0046** (deployed: 0.0358) | PASS |
| A-T4 | Mondrian conformal α=0.05: all classes covered, size ≤ 1.30, singletons ≥ 85% | coverage 0.956, size 1.25, **singletons 0.759**, Normal 0.920 < 0.922 | **FAIL** |
| A-T5 | Multi-label sets ≥ 3× more often on ambiguous inputs | **96.2% vs 14.1% → 6.8×** [6.4, 7.3] | PASS |
| A-T6 | ≤ 2 correct flows deferred per error caught (40.6% catch) | **0.005** (deployed 0.50 floor: 3.06) | PASS |
| B-T1 | Descriptive: unsupported claims in rationales | **98.0%** [97.2, 98.6] contain ≥1 claim the input cannot support; **0.0%** hallucinated values | — |
| B-T2 | Counterfactual comprehensiveness@3 ≥ 2× random | 12.81 vs 4.88 → **2.63×** | PASS |
| B-T3 | Evidence-bound comp@3 ≥ 1.5× rationale-cited; 0 hallucinated; 0 unobservable | **3.58×**, 0, 0 | PASS |
| B-T4 | Gate TreeSHAP top-3 overlap with model evidence ≥ 0.6 | **0.334** | **FAIL** |
| C-T1 | 4-bit: larger conformal sets AND more unobservable claims | sets +0.443 [+0.408, +0.477] ✓; unobservable 0.630 → **0.570** ✗ | **FAIL** |
| D-T1 | GPU↔Pi 5 label agreement ≥ 99%, mean \|Δp\| ≤ 0.02 | **99.3%**, **0.0045** | PASS |
| D-T2 | Distribution scoring ≤ 10% of on-device decision latency | **4.2%** (1.15 s vs 27.6 s) | PASS |

**9 PASS · 3 FAIL · 1 descriptive.**

## Phase 0 — the measurement is the deployed system

Tokenizer reproduces the oracle token ids 300/300; greedy decisions reproduce the published token-id result (macro-F1 **0.9326** vs 0.9321); the class-trie distribution's argmax equals the deployed label on 100% of flows; re-querying gives identical outputs (max |Δp| = 0.0). Checkpoints are SHA-256-identical to the Raspberry Pi's copies.

## Track A — trustworthy uncertainty (test, n = 6,184, accuracy 0.9174)

| Signal | ECE | Brier | AUROC | AURC | Acc. @ 90% coverage |
|---|---|---|---|---|---|
| Verbalized confidence (deployed) | 0.1174 | 0.1118 | 0.7322 | 0.0358 | 0.942 |
| Verbalized + histogram binning *(primary calibrator)* | 0.0094 | 0.0674 | 0.8021 | 0.0250 | 0.942 |
| Intrinsic 15-class probability (no calibration) | 0.0153 | 0.0537 | 0.9328 | 0.0093 | 0.959 |
| Stacked intrinsic + gate *(primary ranking)* | **0.0092** | **0.0248** | **0.9868** | **0.0046** | **0.987** |

- **The model's own token probabilities are already calibrated (ECE 0.015); only the number it writes is not.** The verbalized scale inherited the softened teacher targets used in fine-tuning.
- **Deferral:** at the same ~40% error-catch rate, the deployed floor sends 3.06 correct flows to a human per error caught and defers 13.3% of traffic; the stacked signal sends 0.005 and defers 3.3%.
- **Realistic class mix (n = 600):** ECE 0.0147, AUROC 0.9987; conformal α=0.05 covers every class with size 1.04 and 90.5% singletons (A-T4 would pass on this draw).
- **Seed robustness (3 checkpoints, test_sub1500):** ECE 0.0136 ± 0.0009, AUROC 0.991 ± 0.001, coverage 0.951 ± 0.004.

**Conformal sets explain ambiguity.** 624 test flows (10.1%) carry an input block that appears in training under two or more labels (9,491 such blocks among 469,374). These flows receive a multi-label set 96.2% of the time vs 14.1% otherwise; entropy detects them at AUROC 0.969. Their accuracy is 0.590 vs 0.954 — the uncertainty is concentrated exactly where the input cannot decide. Mean set size peaks on XSS (2.10), Password (1.75), DDoS_HTTP (1.72), Uploading (1.66).

*Why A-T4 failed:* singletons at α=0.05 are 75.9% (target 85%), and Normal covers 0.920 against a lower tolerance of 0.922, with some Normal flows receiving empty sets (1.6% overall).

**Exploratory (not pre-registered):** the strongest single error signal is whether the independent gate backs the model's label (AUROC 0.963). SLM-only signals 0.932, gate-only 0.967, combined 0.987. Dropping the verbalized confidence changes nothing (0.986). Gate and model agree on 85.3% of flows (accuracy 97.7%) and disagree on 14.7% (accuracy 57.0%).

## Track B — faithful explanations

**Claim audit (1,499 full rationales, test_sub1500):**
- Hallucinated values: **0/1,499** — every IP, port and number cited matches the input.
- Claims the input cannot support: **98.0%** [97.2, 98.6], by exhaustive coding of all 54 distinct rationale patterns plus a per-flow URI check for content claims (`out/x3_manual_codes.json`, single coder, released for review). The automatic auditor is a strict lower bound: 63.0% (precision 1.00, recall 0.64).
- Example: **0 of 100 XSS flows show any script content in their input** (`http_full_uri = 0` in 139 of 146 cases), yet their rationales say "payload contains script tags". Unsupported categories: timing/rates 37%, byte sizes 26%, multi-flow patterns 19%, entropy 11%, packet counts 9% (automatic).
- Explanations inherit errors: correct technique cited 100% when the label is right, **6.0%** when it is wrong.

**Counterfactual faithfulness (750 flows, 50/class; in-distribution benign-value interventions):**

| Evidence ranking | comp@3 ↑ | suff@3 ↓ |
|---|---|---|
| Model counterfactual | **12.81** | **1.36** |
| Rationale-cited (+ random fill) | 6.41 | 9.52 |
| Gate TreeSHAP | 5.73 | 10.06 |
| Random fields | 4.88 | 10.63 |
| Rationale-cited only (as written) | 3.58 | — |

Rationales cite on average 1.2 input fields, and 43% cite none. The **evidence-bound explanation** — technique ID + the three decisive fields with their real values + the conformal alternatives when not a singleton — is 3.58× more comprehensive than the rationale's own citations, has 0 hallucinated and 0 unobservable claims by audit, and costs 0.003 ms instead of the 28–42 s an LLM rationale takes on the device.

*Why B-T4 failed:* the gate sees only 6 of the 14 rendered fields (ports, flags, length, MQTT type, ARP opcode), and the model's single most decisive field is visible to the gate in only 28.7% of flows. Where both can see the fields, top-3 agreement is 0.948 (Kendall τ 0.54). SHAP is therefore not a valid on-device substitute for the model's attribution.

## Track C — quantization vs explanation quality (test_sub1500, same flows)

| Model | Acc. | ECE | AUROC | Set size | Singletons |
|---|---|---|---|---|---|
| Q8_0 seed 456 | 0.911 | 0.0125 | 0.991 | 1.22 | 0.775 |
| Q4_K_M seed 456 | 0.861 | 0.0138 | 0.980 | **1.67** | 0.534 |
| bf16 vs Q8_0 (test_sub300) | 0.920 / 0.920 | 0.023 / 0.033 | 0.992 / 0.990 | 1.24 / 1.25 | 0.787 / 0.773 |

- 8-bit is indistinguishable from bf16 on every XAI measure.
- At 4-bit the uncertainty layer stays honest (coverage 0.952, all classes covered, sets +0.44, p ≈ 10⁻⁹⁹) while the **explanation layer breaks** (exploratory, ATT&CK-catalogue checks): **21.3% of 4-bit rationales contradict the model's own label** (Q8: 0%, paired p ≈ 10⁻⁹⁶); 3.4% cite technique IDs that do not exist (Q8: 0); 0.6% hallucinate values (Q8: 0); 393 distinct rationale patterns vs 54; fabricated artefacts include invented hostnames, a SHA-1 hash and a non-existent tool.
- *Why C-T1 failed:* the automatic unobservable rate fell at 4-bit (0.570 vs 0.630, p = 2×10⁻⁶) because the text drifted away from template phrasing into fabrication the lexicon was not built to catch.

## Track D — on the Raspberry Pi 5 (150 flows, 10/class)

Label agreement with the GPU 99.3%, mean |Δp| 0.0045 (max 0.058), on-device accuracy 0.900 (GPU 0.893). The full 15-class distribution costs 1.15 s against a 27.6 s decision. Conformal coverage on these 150 flows is 0.887 [0.826, 0.928]; decomposed on the same flows: GPU with the deployed prefix 0.913 (subset sampling), canonical prefix 0.900 (−0.013, see deviation 4), Pi 0.887 (−0.013, two flows).

## Deviations and disclosures

1. **Corrupted model copies.** Local copies under `D:/Lthesis/models` and `D:/1111111` had correct sizes but wrong bytes and produced garbage; all runs use copies SHA-identical to the Pi (found before pre-registration).
2. **Auditor fixes after first look.** (a) Singular "request" removed as a count unit after a 20-sample manual check found 2 false positives; (b) text copied verbatim from the input is masked before numeric checks. Both only remove false positives; rationale rates were unchanged by (b). The exhaustive manual coding is independent of the auditor.
3. **Evidence-bound wording.** The first version wrote "at the 95% level", which the auditor correctly counted as an unverifiable number; it was rephrased without a numeral. Fields and attribution are unchanged.
4. **Pi prefix.** The Pi runner scored labels at the canonical prefix instead of the deployed grammar prefix, costing about 0.013 coverage on that subset; D-T1/D-T2 are unaffected.
5. **Not pre-registered and labelled exploratory:** stacked-signal ablation, ATT&CK integrity checks, on-device coverage.
6. B2 averages 2 counterfactual draws per single-field intervention and 1 per joint intervention.
7. Generation was interrupted once by a session end and resumed from resumable logs; the determinism gate makes resumed flows identical.

## Reproduce

```
python x0_gates.py                       # Phase 0 gates
python x1_generate.py --jobs A,R,S,Q,F    # generation (GPU, resumable)
python x2_trackA.py --model q8_s456 --val val_cal --tests test,realistic --ambiguity
python x3_audit.py --model q8_s456 --compare q4_s456
python x4_counterfactual.py run && python x4_counterfactual.py analyze
python x5_compare.py
python x6_pi.py export | fetch | analyze
python x7_report.py && python verify_xai.py
```
Figures: `figs/fig1_reliability.png` … `fig7_gpu_vs_pi.png`. Machine-readable: `RESULTS.json`.
