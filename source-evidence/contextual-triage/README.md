# Contextual triage replay

This collection contains the required scientific evidence for the revision-stage contextual triage replay. Read `R4_Supplementary_Report.txt` for the complete methods, exact counts and limitations. An authored referral policy is applied to 6,184 stratified and 600 realistic saved detector occurrences under four hypothetical asset/impact contexts. Every record and empty prediction set is retained. These contexts do not supply observed incident-severity truth.

## Protocol, inputs and results

`PROTOCOL.json` and `PROTOCOL.sha256` preserve the fixed profiles, policy, comparisons and interpretation. `input_records.json` supplies the exact replay inputs; `input_receipt.json` documents their original-source joins, hashes, duplicates and cross-draw overlap. General source paths are retained as provenance, rather than silently rewritten.

`triage_core.py` is the policy implementation. `evaluate_pilot.py` is the portable replay analysis; it needs only this directory and the Python standard library. `results.json` contains the completed aggregate results and implementation checks. `decision_ledger.jsonl` retains all occurrence/profile/mode decisions. Reanalysis writes output files, so use a fresh copy and preserve the released observations. No model weights or new model inference are required.

The set-augmented policy adds 32 main-draw referrals (one attack and 31 Normal) and 21 realistic referrals (all Normal). All additions are empty-set fallback review with Unassigned severity; none is a new Low/Moderate classification. Hypothetical serious impact and unknown context are authored routing conditions. Implementation properties are not measured severity accuracy, and duplicate occurrences are not independent incidents.

## Resident-policy timing

`benchmark_worker.py` retains the standalone benchmark and its fixed input payload. `benchmark_body.py.txt` supplies its readable timing source. `local_timing.json` and `pi_timing.json` are the completed desktop/Pi timing records, including semantic-result hashes. Pi timing uses five warm-up and 30 timed batches of 54,272 resident-policy decisions. It excludes model inference, attribution, input parsing and end-to-end incident handling.

## Release scope and provenance

These scientific files are exact selected members of `Severity_Triage_Evidence.zip`, SHA-256 `82112f014cf04afca793cdc3ef8b65b1344a3c602282708f859b23921e70f372`; this README is new release documentation. The collection is a curated subset, not a replacement archive with all original members. Access/execution administration, duplicate AI-agent verification helpers, preparation helpers and obsolete archive-wide manifests are omitted. No metadata-redacted receipt is needed for this scientific collection. Use the repository publication manifest for the selected file hashes.

Separate AI-agent checks described in the report are not human severity annotations, external peer review or independent experimental replication. The protocol records its own freeze time; the GitHub release establishes present public availability and does not independently certify an earlier timestamp or retrospectively authenticate the original XAI preregistration. Cite a commit-specific repository permalink.
