# Taming transients in analogue solvers

Reproducible numerical experiments for controlling transient amplification in
symmetric positive-definite analogue residual solvers. The project provides
seeded matrix generation, seven bandwidth-allocation policies, mathematical
certificate checks, robustness studies and executable behavioural circuit tests.

## Quick start

Use Python 3.12 and a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
python analyse.py
```

These commands check the supplied generated matrices, allocation constraints,
independent matrix exponentials, Lyapunov residuals, refined time grids and
summary tables. `results/verification.json` records the checks and tolerances.

Install ngspice 47 from its official distribution at
https://ngspice.sourceforge.io/download.html and place `ngspice` on the executable
search path. Its source-release SHA-256 is
`894e649651f1838a14095e5a5439e7d3aa63e87ede14d283173fda4fcdef675f`.
Then recompute the complete suite:

```sh
python run_all.py --workers 6 --ngspice ngspice
```

A full path can be supplied to `--ngspice`. The complete run regenerates numerical
results in `data/` and `results/`, including the recorded allocation and dual-bound
results. Archive a copy of the supplied results before a full run if a side-by-side
comparison is desired. Download access is used for two small structural matrices.

## Contents and execution stages

| File or directory | Purpose |
| --- | --- |
| `config/study.json` | Fixed matrix dimensions, scales, families, seeds and policies |
| `src/core.py` | Matrix generation, row-loading model, allocations, exact propagator and performance metrics |
| `run_study.py` | Parallel 390-case benchmark; completed cases are cached unless `--recompute` is set |
| `analyse.py` | Paired summary statistics, bootstrap intervals and aggregate CSV tables |
| `theory.py` | Exact two-channel transient surface and gain-ratio boundary |
| `engineering.py` | Source downloads and seven-policy tests for two structural matrices |
| `robustness.py` | Gain quantization, coefficient mismatch and two-pole sensitivity |
| `gpu_perturbations.py` | Batched sphere-perturbation trajectories with CPU double-precision cross-checks |
| `spice_check.py` | Nodal netlist generation, ngspice execution and comparison with exact dynamics |
| `certify_optimization.py` | Independently evaluable dual upper bounds for both optimized policies |
| `verify.py` | Regeneration, physical feasibility and independent numerical verification |
| `run_all.py` | Ordered complete reproduction |
| `data/*.npz` | Generated coefficient matrices and the seven corresponding gain vectors |
| `results/cases/*.json` | Case-level metrics, solver statuses and optimization timings |
| `results/dual/*.json` | Case-level rate upper bounds and gaps |
| `results/spice/` | Executable circuit netlists, voltage traces and initial conditions |
| `results/*.csv`, `results/*.json` | Summaries, robustness outcomes, source provenance and validation reports |
| `SHA256SUMS.txt` | File-integrity manifest for the released archive |
| `CITATION.cff`, `LICENSE` | Citation metadata and MIT software licence |

Individual stages can be rerun directly, for example:

```sh
python run_study.py --group main --workers 6 --recompute
python analyse.py
python robustness.py
python gpu_perturbations.py
python spice_check.py --ngspice ngspice
python certify_optimization.py --workers 6 --recompute
python verify.py
```

## Fixed protocol

The main set has 324 matrices: dimensions 8, 16 and 32; three matrix families;
variable scale ratios 1, sqrt(10) and 10; and seeds 1000–1011. The stress set has
54 matrices at scale ratio 100 and seeds 2000–2005. Twelve dimension-64 cases
use scale ratios 1 and 10 and seeds 3000–3001. Every nominal policy uses a total
normalized bandwidth equal to the dimension, with per-channel limits 0.02–8.
The contraction allocation uses margin parameter eta = 0.1.

The seven method keys are:

- `loaded`: equal row bandwidth, including row loading;
- `uniform`: a common dynamical gain across rows;
- `jacobi`: clipped inverse-diagonal allocation;
- `row_norm`: clipped inverse row-norm allocation;
- `blend`: the largest contraction-preserving interpolation from uniform to Jacobi;
- `spectral`: maximization of slowest modal decay under the bandwidth limits;
- `contractive`: spectral-rate maximization with an additional physical-norm contraction constraint.

`rate` is the slowest modal decay rate; `alpha` is the smallest eigenvalue of the
symmetric dynamical part; `gain` is the estimated maximum propagator norm;
`t_1e3` is the last 10^-3 norm crossing. Times are in inverse reference-bandwidth
units except the SPICE traces, which use seconds and volts. `noise_output` and
`noise_input` are stationary per-channel variances with unit noise intensity at
two different injection locations. The total bandwidth is a modelling resource
budget, not a measured power quantity.

## Recorded checks and results

The main-set median settling speedup of `contractive` is 1.384626 relative to
`loaded` and 1.152175 relative to `blend`. There are 133 main-set spectral
allocations with transient gain above 1.00001, compared with zero certified
contraction allocations. All 390 contraction allocations have positive checked
margins. Eight-bit downward logarithmic bandwidth quantization preserves a
positive margin in all 108 tested dimension-32 main systems, retaining a mean
98.6755% of the unquantized rate. These are results for the specified protocol.

Numerical results may differ slightly across solver, BLAS, platform and compiler
versions. Compare continuous metrics with relative tolerance 10^-4; boundary
classifications should also be checked against the stored margin rather than
rounded values. `verify.py` uses tighter tolerances for exact identities within
a single environment. Execution time is environment-dependent. Timings are
recorded as observations, and are not deterministic reproduction targets.

## Computing resources

The recorded benchmark used six CPU workers and one numerical-library thread
per worker. GPU batching automatically selects Metal, CUDA or CPU through
PyTorch. CPU execution supports the same trajectory calculation. The complete
workflow fits within 18 GB host memory. The recorded 390-case benchmark took
approximately 510 seconds; dual certification repeats the optimized solves and
can add a comparable cost. Dimension-64 conic solves dominate the runtime.
GPU batching used 32,768 initial directions per system and 80 sample times,
with double-precision CPU validation of a subset. Results retain the execution
backend and the measured cross-precision discrepancy.

## External data provenance

`engineering.py` obtains the `HB/bcsstk01` and `HB/bcsstk02` Matrix Market archives
from the SuiteSparse Matrix Collection. Their source pages are:

- https://sparse.tamu.edu/HB/bcsstk01
- https://sparse.tamu.edu/HB/bcsstk02

Both are structural stiffness matrices attributed by the collection to J. Lewis.
The source URLs and SHA-256 values are recorded in
`results/engineering_sources.json`; downloads are cached in `data/engineering/`.
They are normalized by the maximum absolute coefficient, without adding
regularization. The collection is described by T. A. Davis and Y. Hu,
“The University of Florida Sparse Matrix Collection,” ACM Transactions on
Mathematical Software 38, 1:1–1:25 (2011), doi:10.1145/2049662.2049663.
The MIT licence applies to this project's software; external data retain their
source terms and attribution.

## Interpretation

The certificate concerns a dominant-pole behavioural model in fixed output
voltage coordinates. The supplied two-pole and noise-location experiments test
its physical limitations. Nodal circuit traces provide an independent simulation
check. The positive voltage boundary is a headroom diagnostic, and random-sphere
exceedance fractions characterize the seeded synthetic perturbations.
