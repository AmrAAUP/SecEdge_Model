# Evidence guide

The manuscript cites this repository as one research-artifact reference. Each collection has a readable entry point and retains the underlying machine-readable records. The table links reviewer requests to those entry points; follow their file links for the primary evidence.

| Reviewer request or paper claim | Evidence | Interpretation |
| --- | --- | --- |
| Verify registration and the thirteen targets | [Registration record](registration.md); [original study](../source-evidence/original-study/) | Exact document and checksum are provided; its recorded date is not independently certified |
| Check the status of Table III | [Original study](../source-evidence/original-study/) | Nine met, three not met, one descriptive; original thresholds and conjunctive conditions remain in force |
| Identify calibration membership and overlap | [Original study](../source-evidence/original-study/) | 3,582 validation calibration rows; fitting and quantile records are reused; duplicate inputs were not excluded |
| Link the 6,184, 600 and 1,500 draws to outputs | [Original study](../source-evidence/original-study/) | Seeds 45, 44 and 2026, respectively; the 1,500 draw is nested in the 6,184 draw |
| Support the 98% rationale finding and exclusion | [Original study](../source-evidence/original-study/); [revision methods](../source-evidence/revision-methods/) | 1,469/1,499; row 2125 excluded for incomplete JSON and null rationale; documented single AI coder |
| Document intervention donors, repetition and sufficiency | [Original study](../source-evidence/original-study/); [Pi donor study](../source-evidence/revision-methods/) | Original marginal pools and score ledger; separate held-out-donor sensitivity experiment |
| Verify GPU–Pi comparisons, quantization and costs | [Original study](../source-evidence/original-study/); [Pi startup](../source-evidence/pi-startup/) | Paired outputs, configuration and timing; label agreement does not establish calibration transfer |
| Reconcile Target 9 with Figure 5 | [Original study](../source-evidence/original-study/) | 12.808/3.579 ≈ 3.58 for cited-only fields; 12.808/6.412 ≈ 2.00 for the filled three-field comparator |
| Investigate severity and practical context | [Contextual triage](../source-evidence/contextual-triage/); [service impact](../source-evidence/service-impact/) | Hypothetical-context replay and a separate measured service-impact forecast study; industrial severity remains outside scope |

## Original study

Read the registration document before the derived audits. Source partitions contain 1,553,434 training, 332,874 validation and 332,893 test rows. Calibration comprises thirteen classes of 250 records, 182 MITM and 150 Fingerprinting. Its reuse and cross-partition matches constrain the interpretation of reported coverage.

The 600-flow realistic draw includes 446 Normal records. The 1,500-flow explanation subset contains 100 records per class from the main draw. Saved row keys, labels and rendered inputs provide the output links. The original support analysis uses 54 rationale patterns; reconstructed row-level joins are identified as reconstructions.

## Later experiments

The Pi donor study uses 30 occurrences representing 28 distinct inputs and five fixed Normal validation donors. Its 1,430 single and 300 joint substitutions evaluate stability separately from the original intervention experiment.

Contextual triage replays saved outputs under hypothetical context. Additional referrals retain unassigned severity when the prediction set is empty. The controlled service experiment instead observes real requests to an isolated HTTP endpoint: 28 episodes, 1,344 total requests and forecasts saved before the later evaluation windows. Functional-monitor agreement is 15/28, compared with 12/28 for the status/latency baseline. Both fail the abrupt-onset and recovery-transition challenges. All 224 reset requests satisfy the authored functional objective.

These experiments are revision-stage additions and do not alter the thirteen original acceptance targets. The repository does not supply a validated industrial severity classifier.

## Reading historical files

Some dated revision and startup reports state that original records were missing. Those statements describe the earlier search state and are superseded by the original-study collection. Original research bytes are retained; public derivatives with access metadata removed are identified in [provenance](provenance.md). A matching checksum establishes integrity, not the truth of a scientific claim or the original date of an event.
