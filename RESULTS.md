---
name: ChronosAttack Codex extension results
type: completed experiment record
---

# GPT-5.6 Luna via Codex CLI

Recorded 2026-09-02 from the completed paper-scale Luna run.

## Result

Luna did not show a statistically significant stateful scheduling-attack effect
in either scenario. The cloud-backup target rate decreased under attack, while
the shipping-partner target rate changed by only 3.3 percentage points.

The secondary stateless cloud-backup comparison showed a large order effect
(10% to 70%, unadjusted Fisher p = 0.0198). It is not the primary stateful
endpoint and does not change the primary conclusion.

## Confirmation experiment

Target choice: A. Primary p-values use the repository's two-sided Fisher test
and Holm correction across the two scenario-level stateful comparisons.

| Scenario | Natural | Attacked | Change | Newcombe 95% CI | Fisher p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Cloud backup | 16/30 (53.3%) | 12/30 (40.0%) | -13.3 pp | [-35.9, 11.3] pp | 0.4379 | 0.8758 |
| Shipping partner | 13/30 (43.3%) | 14/30 (46.7%) | +3.3 pp | [-20.7, 26.8] pp | 1.0000 | 1.0000 |

Secondary stateless comparisons:

| Scenario | Natural | Attacked | Change | Newcombe 95% CI | Unadjusted Fisher p |
|---|---:|---:|---:|---:|---:|
| Cloud backup | 1/10 (10.0%) | 7/10 (70.0%) | +60.0 pp | [17.1, 80.9] pp | 0.0198 |
| Shipping partner | 0/10 (0.0%) | 1/10 (10.0%) | +10.0 pp | [-18.9, 40.4] pp | 1.0000 |

The simultaneous controls selected the target in 1/10 trials (10.0%) for each
scenario.

## Perturbation budget

Observed target-choice rates did not increase monotonically with the number of
pairwise inversions.

| Scenario | 0 inversions | 1 inversion | 2 inversions |
|---|---:|---:|---:|
| Cloud backup | 8/15 (53.3%) | 2/15 (13.3%) | 3/15 (20.0%) |
| Shipping partner | 4/15 (26.7%) | 5/15 (33.3%) | 4/15 (26.7%) |

## Defenses

| Scenario | Barrier target rate | Order-consistency target successes | Defended outcomes | Target permutation votes |
|---|---:|---:|---:|---:|
| Cloud backup | 2/15 (13.3%) | 0/10 (0.0%) | B: 8, abstain: 2 | 17/60 (28.3%) |
| Shipping partner | 1/15 (6.7%) | 0/10 (0.0%) | B: 8, abstain: 2 | 21/60 (35.0%) |

## Completeness and provenance

- Model: `gpt-5.6-luna`; provider label: `codex`; execution mode: `codex_cli_agent`.
- Codex CLI: `0.152.0`; cloned repository revision: `17c4266fc3dc649babf5e33a156f10e257f6e74c`.
- Paper defaults and seed `20260814` were used for all four experiment groups.
- Final data contains 320 scientific result rows: 180 confirmation, 90 budget,
  30 barrier, and 20 order-consistency aggregate trials. Each consistency trial
  contains all six permutations.
- `helper.py` has no diff from the cloned repository revision.

Raw result artifacts:

| Artifact | SHA-256 |
|---|---|
| [`run_manifest.json`](chronos_luna_paper_20260901/run_manifest.json) | `e6f4d67fd53ffd0de99d0193983a3d75014d4b37cc394d77ae54d978097a2a10` |
| [`confirmation_trials.json`](chronos_luna_paper_20260901/codex/confirmation_trials.json) | `19eaddef3730dc03b975708190ad84c18331408a26299d0e1ec0cacf9b92bb58` |
| [`budget_trials.json`](chronos_luna_paper_20260901/codex/budget_trials.json) | `2c5a6e5cf01ba4d6bb201def310cd3e53ca63be4336520daa1cc2f58ca83e12a` |
| [`barrier_trials.json`](chronos_luna_paper_20260901/codex/barrier_trials.json) | `e87638f3f9318b4548066538b5e2f59f2cb951f8d460d71e52733550c0bc6e99` |
| [`order_consistency_trials.json`](chronos_luna_paper_20260901/codex/order_consistency_trials.json) | `a610a22087565ce8f625816f3f5378065b367a7748e7e2e42b92daea76b26b4c` |
