# Provenance and reuse

The public tree is derived from five verified local submission archives. These descriptive collections replace the former appendix labels. The original archives remain unchanged; the mapping is retained for readers of earlier manuscript versions.

| Earlier label | Repository collection | Original archive |
| --- | --- | --- |
| R1 | [revision-methods](../source-evidence/revision-methods/) | `Revision_Evidence.zip` |
| R2 | [pi-startup](../source-evidence/pi-startup/) | `Pi_Startup_Evidence.zip` |
| R3 | [original-study](../source-evidence/original-study/) | `Original_XAI_Evidence.zip` |
| R4 | [contextual-triage](../source-evidence/contextual-triage/) | `Severity_Triage_Evidence.zip` |
| R5 | [service-impact](../source-evidence/service-impact/) | `Operational_Severity_Evidence.zip` |

This curated release contains **158 selected original archive members**, of which **158 retain identical bytes** and **0 are disclosed derivatives**. Selection is limited to the scientific paper's supporting materials. The frozen registration, protocols, scientific inputs, raw outputs, scores and measured times remain unchanged. New navigation guides, notices and publication manifests make the selected evidence usable independently.

[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) maps each selected original member to its current path and records its source and publication hashes. Draft paper outlines, unrelated detector-development material, old search notes, superseded guides, duplicate administrative reports and access-command receipts are excluded. Use the root [MANIFEST_SHA256.json](../MANIFEST_SHA256.json) and [verification script](../verify_manifest.py) for the current public tree. The old ZIP manifests describe a broader archive and are not substituted for this release's manifest.

No private keys, passwords or access tokens were found in the selected evidence. Historical research paths, benchmark addresses and experimental device metadata remain where needed to interpret the records. Publication preparation did not rerun archived launchers or experiments. Independent audits in the records mean separate AI-agent code/data checks, not independent human annotation or external timestamp certification.

## Archive identities

- `Revision_Evidence.zip`: `6df041a194d667c66e25b4a99856dea21a8707806754111bbcf900bfa24063e9`
- `Pi_Startup_Evidence.zip`: `7c1eedc27d5b57866b81ebfc80df237da6df49f31e2d85c5e48af2c343961a63`
- `Original_XAI_Evidence.zip`: `6ed4cf5da090b73676846218c8cc8624bcaa7f5cc9ae29ce6fc1e997903db10d`
- `Severity_Triage_Evidence.zip`: `82112f014cf04afca793cdc3ef8b65b1344a3c602282708f859b23921e70f372`
- `Operational_Severity_Evidence.zip`: `6652070031bcc3d576a2d6c078c2edebd36666785c1fc089f05a82a539f55408`

## Reuse and third-party material

The repository's existing [Apache 2.0 license](../LICENSE) is retained. Dataset material has separate upstream terms: Edge-IIoTset is attributed to Mohamed Amine Ferrag, Othmane Friha, Djallel Hamouda, Leandros Maglaras and Helge Janicke, and its creator's metadata identifies CC BY-NC-SA 4.0. See the [dataset attribution](../third-party-notices/Edge-IIoTset-ATTRIBUTION.md). The bundled tokenizer is associated with the Microsoft Phi-3 model family; the [Microsoft MIT notice](../third-party-notices/Microsoft-Phi-3-MIT-LICENSE.txt) is included. An exact upstream commit for the retained local tokenizer has not been authenticated.

This publication does not relicense third-party data or software, and it contains neither model weights nor the complete source dataset. The retained records support inspection and selected data-only recomputation; a full original training/inference rerun requires the original checkpoints, dataset and compatible runtime.
