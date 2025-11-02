
import hashlib
import json
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def hash_dataset(df: pd.DataFrame) -> str:
    # Stable hash of dataframe content (order independent by sorting columns)
    df_sorted = df.copy()
    df_sorted = df_sorted.reindex(sorted(df_sorted.columns), axis=1)
    data_bytes = df_sorted.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()[:12]

def audit_log(event: dict, fname: str = "audit.jsonl"):
    event = {**event, "timestamp": now_iso()}
    with open(LOG_DIR / fname, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

def bootstrap_ci(values: np.ndarray, func, n_boot: int = 300, alpha: float = 0.05, random_state: int = 42):
    """Generic bootstrap CI for a statistic. Returns (low, high)."""
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    stats = []
    n = len(values)
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        stats.append(func(sample))
    low, high = np.quantile(stats, [alpha/2, 1 - alpha/2])
    return float(low), float(high)
