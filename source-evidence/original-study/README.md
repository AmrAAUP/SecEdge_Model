# Original study evidence

Selected scientific supplementary materials for **Explainable Intrusion Detection with Language Models on GPU-Free Edge Devices**. This collection preserves recovered original study records and the derived membership/item links needed to inspect the paper. The original plan has thirteen targets: nine met, three not met and one descriptive. The [revision methods](../revision-methods/), [Pi startup](../pi-startup/), [contextual triage](../contextual-triage/) and [service-impact study](../service-impact/) are separate collections.

Retained original files have their exact source bytes. The publication manifest records the selected files and their hashes. Paper outlines, administrative launch/progress files, duplicate plot rasters, verbose server logs and duplicate internal audit narratives are not part of this curated collection. Original Python scripts are supplied as source; reading this evidence does not require running models or contacting a device.

## Registration and reported outcomes

- [PREREG_XAI.json](original_xai2026/PREREG_XAI.json) and [original checksum sidecar](original_xai2026/PREREG_XAI.sha256).
- [Machine-readable outcomes](original_xai2026/RESULTS.json) and [readable outcomes](original_xai2026/RESULTS.md).

The plan's full SHA-256 is `75bba6dbc5bf8fe65482efd5699a03d1e1bc7334224d02a893342007b46e2684`. It records 11 September 2026, 00:55:31 +01:00; this is documentary evidence, not an independently certified timestamp. Later repository publication does not certify the plan's earlier chronology. Declared criteria, post-inspection refinements and exploratory checks must remain distinguished.

## Frozen inputs, populations and overlap

[The frozen-input directory](frozen_inputs/) preserves the 6,184-row main and 600-row realistic CSVs and configuration. [The draw directory](original_xai2026/out/draws/) contains original index files and the 3,582-row calibration cache: thirteen classes of 250 records, 182 MITM and 150 Fingerprinting. The 1,500 explanation subset is nested in the main draw; the 300 precision subset is nested in the 1,500 subset.

[Derived calibration membership](independent_review/original_calibration_membership_evidence_audit.csv), [calibration/draw provenance](independent_review/calibration_draw_provenance_evidence_audit.json) and [training-overlap records](independent_review/training_overlap_provenance_evidence_audit.json) retain source/value/rendered-input bindings needed for the reported overlap counts. These derived checks are separate from original records. Calibration fitting and conformal quantiles reuse records; repeated inputs limit coverage interpretation.

## Calibration and prediction sets

[Original output records](original_xai2026/out/) include keyed `x1_*_dec.jsonl` predictions, `x2_trackA_*.json` fitted-calibration results and numeric `x2_*_scores.npz` score/set arrays. [x1_generate.py](original_xai2026/x1_generate.py) and [x2_trackA.py](original_xai2026/x2_trackA.py) record producer and analysis logic. Numeric NPZ records can be read without pickle loading.

## Annotations and exclusion

[The original 1,500 full outputs](original_xai2026/out/x1_q8_s456_test_sub1500_full.jsonl), [54 pattern forms](original_xai2026/out/x3_patterns.json), [saved codes](original_xai2026/out/x3_manual_codes.json), [coding reference](original_xai2026/out/x3_manual_reference.json) and [automatic row audit](original_xai2026/out/x3_audit_q8_s456_rows.jsonl) supply the documented annotation evidence.

[Derived item-level links](independent_review/faithfulness_annotation_record_links_20261008.jsonl) reconstruct the pattern-to-record mapping. They yield 1,469 unsupported cases among 1,499 parsed rationales. Row 2125 is the sole incomplete-JSON exclusion. The saved coding identifies one AI assistant coder; these records do not establish independent second-human agreement.

## Interventions, sufficiency and comparator arithmetic

[Original intervention source](original_xai2026/x4_counterfactual.py), [Normal marginal pools](original_xai2026/out/normal_pools.json), [750-row effect ledger](original_xai2026/out/x4_counterfactual.jsonl), [aggregate scores](original_xai2026/out/x4_trackB.json), [constructed text](original_xai2026/out/x4_evidence_explanations.jsonl) and [scoring/cache progression](original_xai2026/out/x4_run.log) support this analysis.

Target 9 compares ranked fields with genuine cited-only fields (3.58x). Figure 5 uses rationale lists filled to three fields (2.00x). Both denominators are retained. Marginal pools do not retain coherent donor rows or guarantee packet validity; averaged single-field effects cannot recover original draw-to-draw stability. The separate revision-methods collection supplies the five-donor study. Approximately 0.003 ms measures formatting after attribution, excluding model/intervention calls.

## Paired devices, precision and costs

[GPU reference](original_xai2026/out/x6_gpu_reference.jsonl), [Pi bundle](original_xai2026/out/pi_bundle.json), [Pi outputs](original_xai2026/out/pi_results.jsonl), [device summary](original_xai2026/out/x6_trackD.json), [retained Pi worker](original_xai2026/out/xai_pi_run.py) and [checkpoint hashes](original_xai2026/out/model_hashes.json) preserve the 150-row comparison. Keyed Q8/Q4 1,500-row and Q8/BF16 300-row outputs, associated validation records, [precision comparison](original_xai2026/out/x5_trackC.json) and original producer sources preserve precision effects and per-record timings.

Label agreement is 149/150, while transferred-calibrator coverage is 133/150. Pi scoring overhead is relative to constrained 48-token decisions, excluding attribution/full rationale generation. Each precision uses separate calibration. The retained producer configures BF16 with 20 GPU layers and Q8/Q4 with 99, so timing does not isolate precision alone. This records configured producer logic; independently verified physical GPU identity or effective offload was not recorded. Complete original end-to-end Pi attribution cost is not supplied.

## Analysis and verification source

[Original analysis modules](original_xai2026/) and the retained independent scripts for [calibration/draws](independent_review/calibration_draw_audit_evidence.py), [training overlap](independent_review/training_overlap_evidence_audit.py), [metric arithmetic](independent_review/metric_arithmetic_evidence_audit.py) and [GPU/precision/cost joins](independent_review/audit_gpu_quantization_locate_original.py) make the saved results inspectable. The original figures are retained; duplicate paper-layout rasters are omitted.

## Upstream data and notices

Benchmark timestamps, testbed addresses and protocol/payload fields are research dataset records. Historical local source/model paths and an internal Pi destination are configuration/provenance metadata, not access credentials. Model weights, SSH keys and unrelated private files are excluded.

The retained CSVs are Edge-IIoTset subsets. Its author's [primary metadata](https://www.kaggle.com/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/metadata) identifies CC BY-NC-SA 4.0. [Third-party notices](../../third-party-notices/) retain dataset attribution/terms and Microsoft's exact MIT notice for the bundled Phi-3 tokenizer. Dataset and tokenizer material are not relicensed by the repository's code license. An exact original upstream tokenizer commit is not authenticated.
