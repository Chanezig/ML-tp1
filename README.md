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

## What is not reproduced yet

The lock file and the hash fix the package selections and the input bytes. They do **not** fix the OS, hardware or security policies, the label quality, or any training result: no model is trained in TP01.
