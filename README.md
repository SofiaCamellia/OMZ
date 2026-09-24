# OMZ: Attainable-Set Evaluation

Minimal reference implementation for **Evaluating Probabilistic Threshold-Run Predictions from Partial Observations**, extracted and adapted from the research code. Includes exact marginal and joint count-span attainable sets, envelope/global-feasibility baselines, prediction-set assessments, and common-truth Brier/CRPS score comparisons.

## Quick start

Requires Python 3.10+ and NumPy. Run from this directory; no GPU or external data is needed.

```bash
python -m pip install -r requirements.txt
python -B example.py
python -B -m unittest -v test_omz.py
```

## Files

- `omz.py`: core algorithms and evaluation interfaces.
- `example.py`: runnable synthetic demonstration with assertions.
- `test_omz.py` and `_oracle.py`: correctness checks and independent small-grid enumeration. Keep both files to run the tests.
- `requirements.txt`: NumPy dependency.

## Key conventions

- Inputs use a regular grid. `allowed(y, valid, ...)` requires an explicit Boolean observation mask; missing values may be NaN.
- A low-oxygen node satisfies `y < threshold` (default: 60 micromol/kg). A qualifying layer is a maximal low-oxygen run of at least `k` nodes (default: 3).
- Endpoints are `(E, A, B, K, H)`: occurrence, first/last position, layer count, and summed layer span. `A` and `B` use grid indices; category `J` denotes absence, not a depth.
- **H is returned in grid steps, not metres.** A run of `r` nodes contributes `r-1` steps. Multiply by grid spacing to obtain metres; `k=3` with 10 m spacing gives a 20 m minimum span.
- `classify` returns `0` (inclusion), `1` (exclusion), or `2` (indeterminate). Keep predictions fixed when comparing attainable-set alternatives. See `example.py` for joint-set encoding and score comparisons.

## Scope

The example illustrates the method on synthetic inputs; it does **not** reproduce the paper's numerical tables. This release does not include observational datasets, model training, checkpoints, frozen predictions, or the full experimental pipeline. Certificates apply to the specified finite grid and observation constraints, not to model calibration or unobserved continuous-profile geometry.
