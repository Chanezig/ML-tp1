# churnlab

Reproducible churn prediction project (ESILV – Machine Learning Operations).

## Business question

A telecommunications company wants to identify customers who may leave its service.
**Question:** given a customer's current profile and contract, will this customer churn?

- **Target column:** `Churn` (`Yes` / `No`), a binary classification target.
- `Churn` is never used as a model input (target leakage), and `customerID` is an identifier, not a feature.
- Planned baseline inputs (later TPs): `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`.

## Environment setup

Requirements: **Python 3.11** (pinned in `.python-version`; `requires-python = ">=3.11"`), Git and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run python -c "import churnlab; print(churnlab.__version__)"   # expected: 0.1.0
uv run python scripts/record_environment.py                       # writes evidence/tp01/environment.json
```

`uv sync --locked` installs exactly the versions recorded in `uv.lock` and fails if the lock no longer matches `pyproject.toml`. If it fails, investigate the mismatch; do not regenerate the lock silently.

> **Windows note:** if `import pandas` fails with *"DLL load failed … an application control policy blocked this file"*, Windows Smart App Control is blocking a recently released wheel. This is why `uv.lock` pins `pandas 3.0.5` rather than `3.0.6`.

## Dataset

**IBM Telco Customer Churn**, a CSV of 7,043 rows and 21 columns.

| Item | Value |
|---|---|
| Source repository | <https://github.com/IBM/telco-customer-churn-on-icp4d> |
| Pinned commit | `1fd6fd70906479be1712e64cfc0ea89f58466629` |
| File URL | [Telco-Customer-Churn.csv](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv) |
| Local path | `data/raw/telco.csv` (not committed, see `.gitignore`) |
| SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |

Get and verify the data, then inspect it:

```sh
uv run python scripts/download_data.py   # downloads the CSV if absent, checks the hash, writes data/raw/telco.source.json
uv run python scripts/inspect_telco.py   # checks the hash again, writes reports/tp01_data_overview.json
```

Expected inspection results: 7,043 rows, 21 columns, `Churn = No` 5,174 / `Yes` 1,869 (≈ 26.54 % churn), 0 duplicate `customerID`, 11 whitespace-only `TotalCharges` values.
The scripts stop with an error if the hash does not match. In that case, get the original snapshot again; do not disable the check.
Never edit `data/raw/telco.csv`. Cleaning happens in later TPs and writes to `data/processed/`.

### What this dataset is (and is not)

This is a **single customer snapshot**: one row per customer, with the churn label observed at one point in time.
It is **not** an event-time log (no dated history of customer actions, so we cannot rebuild what was known *before* a customer left), and it is **not** an intervention dataset (no randomized retention offer or treatment assignment, so we cannot estimate the causal effect of an action).
Models trained on it predict association with churn, not the effect of a retention campaign. Do not invent event histories or treatments and present them as IBM data.

## Project layout

| Path | Purpose |
|---|---|
| `pyproject.toml` | Project metadata, Python constraint and declared dependencies |
| `uv.lock` | Exact resolved versions of all direct and transitive packages |
| `src/churnlab/` | Importable project code (data, training, serving) |
| `scripts/` | Executable workflow commands (download, inspection, environment record) |
| `tests/` | Small behaviour checks |
| `data/raw/` | Original input snapshot (immutable) and its source metadata `telco.source.json` |
| `data/processed/` | Generated train/test splits (later TPs, not committed) |
| `artifacts/` | Generated model files (later TPs, not committed) |
| `reports/` | Generated reports (not committed; copy small evidence to `evidence/`) |
| `evidence/` | Submitted, non-sensitive evidence per TP |
| `k8s/`, `pipelines/` | Deployment and orchestration (later TPs) |

## Training pipeline (TP02)

### Run it

```sh
uv sync --locked                          # includes pytest (dev dependency)
uv run python scripts/download_data.py    # only if data/raw/telco.csv is absent
uv run python scripts/first_model.py      # milestone 1: 2-feature model vs majority baseline
uv run python scripts/train.py            # compare candidates on validation, save the selected pipeline (outputs in reports/tp02/)
uv run pytest -v                          # reliability checks (splits, leakage, missing values, reload)
uv run python scripts/predict.py --input evidence/tp02/sample_batch.csv   # inference only, no fit
uv run python scripts/evaluate_test.py    # ONCE, after selection: held-out test estimate
```

All settings (seed, split sizes, threshold, candidates and the selection rule) are in [configs/tp02.json](configs/tp02.json). They were fixed **before** looking at the results.

`train.py` writes its comparison table, metadata copy and sample batch to `reports/tp02/`, which Git ignores, so running it leaves the working tree clean. The submitted evidence in `evidence/tp02/` was produced with `--evidence-dir evidence/tp02` at commit `cbd7081`. You should get the same artifact SHA-256 (`0cde2a53…`), the same metrics, and different fit times.

### Task and data

- Positive class: `Churn = Yes` → `1` (`No` → `0`). `customerID` is used only to identify customers and check splits; neither it nor `Churn` is a model input.
- Split by customer, stratified on `Churn`, seed `42`: train 60 % (4,225) / validation 20 % (1,409) / test 20 % (1,409). The allocation is saved in `evidence/tp02/split_allocation.csv`.
- Validation is used to compare candidates. The test set stays reserved and is evaluated once, after the choice is made.

### Input schema

| Feature | Type expected at inference | Rule |
|---|---|---|
| `tenure`, `MonthlyCharges`, `TotalCharges` | number (or numeric text) | blank / missing → NaN, then imputed with the **training** median; any other text (e.g. `"abc"`) → `ValueError` |
| `Contract`, `InternetService`, `PaymentMethod` | text | blank / missing → `"missing"`; a category not seen in training is encoded as all zeros (ignored) |

A missing column raises `ValueError`. Parsing is done by `churnlab.data.prepare_features` before the pipeline, identically for training and inference.

### Pipeline

`Pipeline(preprocessor → classifier)` built by `churnlab.pipeline.build_pipeline` (unfitted; it never reads or fits data):

- numeric branch: `SimpleImputer(median)` → `StandardScaler`
- categorical branch: `SimpleImputer(constant "missing")` → `OneHotEncoder(handle_unknown="ignore")`

A single `fit` on the training features learns every statistic; validation and new data only call `predict` / `predict_proba`.

### Comparison and selection (validation set)

Selection metric chosen in advance: **average precision** of the class-1 probability. If a simpler model is within 0.01 of the best, the simpler model wins (logistic regression < decision tree < random forest). Threshold metrics use 0.5.

| Model | Avg precision | ROC AUC | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.265 | 0.500 | 0.000 | 0.000 | 0.000 | 0.735 |
| Logistic regression | 0.647 | 0.849 | 0.664 | 0.529 | 0.589 | 0.804 |
| Decision tree (depth 5) | 0.617 | 0.838 | 0.636 | 0.602 | 0.618 | 0.803 |
| **Random forest** (300 trees, depth 8) | **0.664** | 0.853 | 0.653 | 0.484 | 0.556 | 0.795 |

Selected: **random forest**. It is 0.017 above logistic regression, which is outside the 0.01 tolerance. Full table: `evidence/tp02/comparison.csv`. Held-out test (evaluated once): AP 0.635, recall 0.479, accuracy 0.782.

### Artifact

- `artifacts/churn_pipeline.joblib`: the **fitted preprocessing + classifier** together (not in Git).
- `artifacts/churn_pipeline.metadata.json`, also copied to `evidence/tp02/model_metadata.json`: artifact SHA-256, source hash, seed, allocation hash, feature roles, preprocessing settings, parameters, validation metrics, threshold, Git commit and library versions.
- `scripts/predict.py` checks the artifact hash against the metadata before loading. joblib uses pickle: **only load artifacts you trust**, with the same library versions (see the metadata).

## What is not reproduced yet

The lock file and the hash fix the package selections and the input bytes. They do **not** fix the OS, hardware or security policies, or the label quality.
Training is deterministic for a given seed and environment (same artifact SHA-256 across runs). Random forest probabilities can still differ at about 1e-16 because of multithreaded summation, so reload checks use a 1e-12 tolerance.
Probabilities are not calibrated: a score of 0.8 is not a measured 80 % risk. Runs are not tracked in an experiment store yet (next TP).
