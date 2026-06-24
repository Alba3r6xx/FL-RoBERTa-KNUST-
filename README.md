# FL-RoBERTa-KNUST-

Starter codebase for the KNUST FL-RoBERTa research project on AI-generated academic text detection.

## Included starter components

- `src/fl_roberta/config.py` – experiment defaults from the proposal
- `src/fl_roberta/losses.py` – focal loss implementation for FL-RoBERTa
- `src/fl_roberta/data.py` – preprocessing + stratified split helpers
- `scripts/train_baselines.py` – logistic regression baseline runner
- `scripts/train_fl_roberta.py` – FL-RoBERTa training entrypoint scaffold

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
python scripts/train_baselines.py --data /absolute/path/to/data.csv --text-col text --label-col label
python scripts/train_fl_roberta.py --data /absolute/path/to/data.csv --text-col text --label-col label
```
