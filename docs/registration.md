# Registration record

The original [PREREG_XAI.json](../source-evidence/original-study/original_xai2026/PREREG_XAI.json) declares the XAI study plan and thirteen targets. Its recorded timestamp is **11 September 2026 at 00:55:31 +01:00**. Its status describes registration before test scoring; that status is a statement in the retained document.

The full SHA-256, matching the retained [checksum sidecar](../source-evidence/original-study/original_xai2026/PREREG_XAI.sha256), is:

```text
75bba6dbc5bf8fe65482efd5699a03d1e1bc7334224d02a893342007b46e2684
```

Download the file without resaving it and compute its SHA-256. For example, from the repository root:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'source-evidence/original-study/original_xai2026/PREREG_XAI.json'
```

On a system with `sha256sum`, the equivalent check is:

```sh
sha256sum source-evidence/original-study/original_xai2026/PREREG_XAI.json
```

A matching digest verifies the exact document bytes. The accompanying frozen-input hashes and [draw manifest](../source-evidence/original-study/original_xai2026/out/draws/manifest.json) connect the protocol to retained inputs. [RESULTS.md](../source-evidence/original-study/original_xai2026/RESULTS.md) and [RESULTS.json](../source-evidence/original-study/original_xai2026/RESULTS.json) report outcomes against the original definitions: nine met, three not met and one descriptive.

## What the dates establish

The file supplies documentary registration evidence. Its recorded timestamp and status are not independently certified. GitHub records publication of this recovered evidence; publication in October 2026 does not itself prove that the plan was fixed before the September evaluation. The manuscript states this distinction explicitly.

Post-inspection auditor changes and exploratory analyses are distinguished from the original declared plan. The [donor-sensitivity study](../source-evidence/revision-methods/), [contextual triage replay](../source-evidence/contextual-triage/) and [controlled service experiment](../source-evidence/service-impact/) are later revision-stage work with their own records. They add no original acceptance targets.
