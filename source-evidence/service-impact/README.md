# Controlled Pi service-impact forecasting

This collection contains the required scientific evidence for one fixed revision-stage experiment on a disposable loopback-only Pi service. Read `R5_Supplementary_Report.txt` for the complete protocol, condition-level results and confusion matrices. The experiment measures actual HTTP status, nonce/digest correctness and request-start-to-completion time. It forecasts authored service-SLO impact tiers for distinct later requests; it does not validate severity learned by the LLM, industrial harm or human urgency.

## Fixed protocol and observed outcomes

`PROTOCOL.json` and `PROTOCOL.sha256` preserve the thresholds, randomized order, workload and temporal boundaries. `service_worker.py` is the standalone standard-library worker; `service_worker_body.py.txt` provides its readable source body. The worker creates an ephemeral loopback endpoint only. Running it would produce a separately labelled new experiment, rather than reproduce the saved runtime.

`service_run.jsonl` retains all 28 randomized episodes and 1,344 observed requests: 16 earlier monitor, 24 distinct later client and eight healthy reset requests per episode. It includes the protocol digest, run-start timestamp, forecasts committed before every later request, measured response fields, episode outcomes and completed service closure. The raw outcome record, not the configured fault, supplies the later reference tier. Its completion event records a stopped service thread. `service_cleanup_readonly_check.json` records the subsequent absence of the disposable process/listener; `device_context.json` preserves the captured device environment.

`analysis.json` contains the complete metrics and episode/condition details. `analyze_service.py` is portable data-only analysis using the standard library; it checks raw success fields, forecasts and later tiers against the fixed protocol. It writes analysis output, so use a fresh copy and preserve the released measurements. No model inference is involved.

## Results and interpretation

Functional-monitor tier agreement is 15/28 versus 12/28 for status/latency alone; stationary agreement is 15/20 versus 12/20. Both fail all eight onset/recovery transitions. High-tier detection is 6/12 versus 4/12, so deployment adequacy is not established. All 224 reset requests pass. Request timing begins at actual request-thread start and excludes pre-dispatch scheduling lag.

The authored 250 ms objective and four tiers concern this endpoint. Twenty-eight episodes, not 1,344 requests, are tier-forecast units. Four repeats per condition on one device/service do not establish industrial generalization. Separate AI-agent code/data checks described in the report are not human annotations, external peer review or independent experimental replication. No original acceptance target or contextual-triage policy was changed.

## Release scope and provenance

These scientific files are exact selected members of `Operational_Severity_Evidence.zip`, SHA-256 `6652070031bcc3d576a2d6c078c2edebd36666785c1fc089f05a82a539f55408`; this README is new release documentation. The collection is a curated subset. Private access/execution administration, duplicate agent verification scripts/reports, preparation helpers and obsolete archive-wide manifests are omitted. No metadata-redacted execution receipt is needed for this scientific collection. Use the repository publication manifest for the selected file hashes.

The protocol and raw event clocks document the retained freeze/run sequence and temporal commitments; they are not independent timestamp certification. GitHub publication establishes current public availability, not the original XAI preregistration date. Cite a commit-specific repository permalink.
