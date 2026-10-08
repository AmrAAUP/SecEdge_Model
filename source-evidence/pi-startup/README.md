# Retained Pi XAI workflow and component costs

This collection provides the retained Pi source, inputs, outputs and configuration
needed to inspect the XAI decision workflow. Start with the
[current evidence guide](../../docs/evidence-guide.md). The original registration
and paired GPU/Pi/precision evidence are in the
[original-study collection](../original-study/).

## Decision and class-probability workflow

Read `original_sources/pi5_h3_measure/xai_pi_run.py`, `xai_pi_bundle.json` and
`out_pi/xai_pi_results.jsonl` together. They bind 150 saved inputs to constrained
decisions and class-trie scoring, with four threads, a 2,048-token context and a
48-token generation cap. The accompanying log and configuration retain the
runtime evidence. Trie scoring is a separate cached-prompt component: its
overhead excludes full attribution, explanation generation and end-to-end cost.

The separate `xai_ondevice.py` and `out_pi/xai_ondevice.jsonl` preserve the
full-output confirmation workflow. Its timer, schema parse flag and draw must
not replace the decision-only workflow's timing boundary. Both workers,
`secedge_lib.py`, configuration, grammars and the two completed logs are retained
unchanged. Source code is archival and expects the original data, model and
runtime; model checkpoints are not included and the source was not executed to
assemble this publication collection.

## Binding and environment records

`metadata/h3_output_binding_manifest.json` provides exact output, grammar and
preflight source hashes with capture metadata. `out/01_preflight.json` records a
retained earlier runtime observation. `metadata/current_pi_hardware_runtime_readonly.json`
is a later hardware observation; it does not establish the hardware or runtime
of every historical result. Filesystem times and recorded configuration are
provenance evidence rather than independently certified event timestamps.

## Curated publication scope

Earlier `fulltest` and `paper/paper_q1` detector/deployment projects, generic
installation/launcher guides, archive administration and superseded missing-record
search reports are omitted. They are not needed to inspect this XAI workflow.
Included scientific files retain their original bytes; the repository publication
manifest covers their paths and hashes. Original machine and dataset paths in
these records provide context rather than portable execution instructions.

Use a commit-specific repository permalink when citing this collection.
