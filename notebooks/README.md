# EPL Outcome Predictor — Exploration

Run the pipeline first:

```bash
python run_pipeline.py
```

Then use this notebook for ad-hoc EDA / Optuna sweeps.

```python
import pandas as pd
from pathlib import Path

root = Path("..")
feats = pd.read_parquet(root / "data/processed/features.parquet")
preds = pd.read_parquet(root / "data/processed/test_predictions.parquet")
feats["outcome"].value_counts(normalize=True), preds.head()
```
