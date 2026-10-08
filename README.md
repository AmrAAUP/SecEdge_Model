# SecEdge Model research evidence

Research artifacts accompanying **Explainable Intrusion Detection with Language Models on GPU-Free Edge Devices**, by Amr Jamal Atrash, Majdi Owda and Amani Yousef Owda.

This repository connects the paper's claims to registration records, sampled inputs, saved outputs, analysis code and later Raspberry Pi experiments. Start with the [evidence guide](docs/evidence-guide.md), which maps each reviewer request to the relevant records. The original thirteen targets retain nine met outcomes, three not met outcomes and one descriptive outcome.

## Evidence collections

| Collection | Contents |
| --- | --- |
| [Original study](source-evidence/original-study/) | Registration bytes and hashes, calibration membership, test draws, annotations, interventions, paired device and precision outputs |
| [Revision methods](source-evidence/revision-methods/) | Documented audit methods and the separate Pi donor-sensitivity experiment |
| [Pi startup and costs](source-evidence/pi-startup/) | Retained startup workers, configuration and component measurements |
| [Contextual triage](source-evidence/contextual-triage/) | Exploratory replay under hypothetical asset and impact contexts |
| [Service impact](source-evidence/service-impact/) | Prospective protocol, forecasts and measured requests from the isolated Pi HTTP service |

The collection names replace the earlier R1–R5 appendix labels. The [provenance record](docs/provenance.md) retains the mapping so older manuscript and archive references remain traceable. Historical reports describe what was known when written; the original-study collection supplies the records recovered later.

## Registration and integrity

`PREREG_XAI.json` records **11 September 2026 at 00:55:31 +01:00**. Its original SHA-256 is:

```text
75bba6dbc5bf8fe65482efd5699a03d1e1bc7334224d02a893342007b46e2684
```

See the [registration record](docs/registration.md) for the exact file and checksum sidecar. A matching hash verifies file identity. The recorded date and status are documentary evidence; this repository's later publication does not independently certify that the plan preceded the experiment.

[MANIFEST_SHA256.json](MANIFEST_SHA256.json) lists the published files and their hashes. From the repository root, run `python verify_manifest.py` to check the downloaded bytes. This check does not run experiments or contact a device. Download the entire repository to inspect large records that GitHub does not display inline.

## Scope of the evidence

The original detector predicts attack types. Calibration fitting and conformal quantiles reuse records, and repeated inputs occur across partitions; the paper reports empirical coverage under these conditions. The 98% rationale result uses a documented single AI-coder protocol. Intervention measurements assess input sensitivity without guaranteeing physical packet validity. Component timings exclude the full attribution cost.

The revision-stage service experiment measures short-horizon forecasts of an authored service objective on one isolated Pi endpoint. It does not establish industrial harm levels or validate the language model as an incident-severity classifier. All transition failures and adverse outcomes remain in the records.

## Citation

A. J. Atrash, M. Owda, and A. Y. Owda, “SecEdge Model research evidence,” version 2026.10.08, GitHub, 2026. [Online]. Available: https://github.com/AmrAAUP/SecEdge_Model

[CITATION.cff](CITATION.cff) provides citation metadata. Cite the specific commit used for an analysis; a branch can change. No DOI or independent registration timestamp is claimed.

## Reuse

The repository retains its existing [Apache 2.0 license](LICENSE). Original dataset and third-party software rights remain with their respective owners; this evidence release does not relicense them. See [provenance and reuse](docs/provenance.md). Model checkpoints and the complete source dataset are not included.
