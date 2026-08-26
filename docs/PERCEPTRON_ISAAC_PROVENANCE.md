# Perceptron Isaac provenance

This document records the source lineage and release boundaries of the
`perceptron_isaac` integration. It intentionally excludes workstation paths,
cluster jobs, private object-store locations, and mutable validation results.

## LeRobot lineage

The native integration was forward-ported onto LeRobot commit
`1fe58f2d3afe0e7c46e86fee03de2e4122fbe9a1`. The port was reviewed file by
file; it was not produced by merging an older fork wholesale.

| Area                          | Reviewed source                                            | Purpose                                                                  |
| ----------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------ |
| Policy, model, and processors | internal history `bf07dd45..30a0f081`, snapshot `1d489ff6` | Behavioral source for the native LeRobot implementation                  |
| Training semantics            | internal snapshot `44bbccdd`                               | Semantic oracle; the shared LeRobot trainer remains authoritative        |
| YAM hardware support          | historical fork snapshot `e137f601`                        | Hardware interface source, followed by safety hardening                  |
| Feetech connection handling   | internal snapshot `30a0f081`                               | Retry and connection behavior reconciled with current LeRobot            |
| Async transport behavior      | internal snapshot `1d489ff6`                               | Source for the native async-inference integration                        |
| SO100/SO101 serving behavior  | `irenegracekp/molmoact2-so101@9819f6a`                     | Observation-timestamped chunks, temporal ensembling, and motion limiting |

Internal commit identifiers are retained so maintainers can audit the port.
They are provenance references, not runtime dependencies.

## External components

- FAST tokenizer and processor: `physical-intelligence/fast` at
  `ec4d7aa71691cac0b8bed6942be45684db2110f4`. The reviewed source tree has
  SHA-256
  `127eb029e5acb8242c7bc2c8efcea6c5dd6ffb565353b5c689b8c8ac55f33a8a`.
- YAM i2rt support: `whats2000/i2rt` at
  `493f0990459049d4fa46898e7d67e4fb55ba76b5`.
- Qwen 3.5 model components are loaded through Transformers. The lockfile is
  the source of truth for the released Transformers version.
- mharmony and TensorStream are separate projects. The public release must
  consume their published packages through normal imports and pinned
  dependencies; sibling-checkout `PYTHONPATH` instructions are not a
  supported installation path.

The dependency and licensing review for each external component must be
complete before a public tag. If a component is not yet published, the release
must not silently fall back to a developer checkout.

## Checkpoint and dataset provenance

Weights, datasets, and conversion outputs have their own provenance and
licenses. Each published model card must record:

- the source checkpoint and immutable revision;
- the conversion tool and revision;
- hashes for the exported contract and identity sidecars;
- the selected normalization profile;
- training datasets and their licenses; and
- reproducible evaluation commands and results.

Local checkpoints and generated reports do not belong in this repository.
`checkpoint/`, `reports/`, and cluster launch files are ignored for that
reason.

## Release claims

Parity, benchmark, and hardware-acceptance claims should be made only from
reproducible public artifacts. Historical job IDs, machine-local paths, and
one-off investigation reports are deliberately not part of this ledger.
