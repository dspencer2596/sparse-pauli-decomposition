# Pauli decomposition

Python implementation and benchmarking code for a randomized sparse Pauli decomposition algorithm.

The main script works by creating random linear combinations of Pauli strings, applying the Pauli decomposition algorithm to recover the Pauli strings and their coefficients, and records runtime/query-count behavior. It can optionally compare against PennyLane's dense Pauli decomposition on small instances.

## What the code does

The implementation represents Pauli labels by bit strings `(x, z)` and recovers coefficients in a sparse expansion

```text
M = sum_{x,z} alpha_{x,z} P_{x,z}
```

for `k`-sparse instances. The main recovery routine combines:

- randomized active-`x` support discovery from row queries,
- certification for slices containing a unique `z`,
- folded sparse Walsh-Hadamard decoding for degenerate `x`-slices,
- final residual checks and coefficient-error reporting.

The script also includes utilities for generating random instances, comparing recovered coefficients against ground truth, running multi-trial benchmarks, saving CSV/cache files, and plotting runtime/query scaling.

## Requirements

Core requirements:

```text
numpy
pandas
matplotlib
```

Optional requirements for the PennyLane baseline:

```text
pennylane
scipy
```

If you do not want the PennyLane comparison, set `run_pennylane = False` inside `main()` before running the benchmark.

## Outputs

A benchmark run may create files such as:

```text
figures/runtime.pdf
figures/queries.pdf
figures/simulation_results_mixed_pennylane.csv
figures/simulation_results_mixed_pennylane.pkl
```
