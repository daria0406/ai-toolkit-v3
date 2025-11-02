
import io
import numpy as np
import pandas as pd

REQUIRED_COLS = {"label", "protected_attr"}

def validate_schema(df: pd.DataFrame):
    missing = REQUIRED_COLS - set(df.columns)
    errs = []
    if missing:
        errs.append(f"Missing required columns: {sorted(list(missing))}")
    if "label" in df.columns and not set(pd.unique(df["label"]).tolist()) <= {0,1}:
        errs.append("Column 'label' must be binary (0/1).")
    if "protected_attr" in df.columns and df["protected_attr"].isna().any():
        errs.append("Column 'protected_attr' contains missing values.")
    return errs

def generate_sample(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = rng.choice(["A","B"], size=n, p=[0.6, 0.4])
    # features
    x1 = rng.normal(0, 1, size=n) + (groups == "B") * 0.4
    x2 = rng.normal(0, 1.2, size=n) + (groups == "A") * -0.2
    # latent score with group skew
    score = 1/(1+np.exp(-(0.8*x1 + 0.6*x2 + (groups=="B")*0.5 - 0.1)))
    # threshold with mild bias (higher threshold for B)
    base_thresh = 0.5
    th = np.where(groups=="B", base_thresh+0.05, base_thresh)
    y_pred = (score >= th).astype(int)
    # noisy label generation
    label = (score + rng.normal(0, 0.15, size=n) > 0.5).astype(int)
    df = pd.DataFrame({
        "feature_x1": x1,
        "feature_x2": x2,
        "score": score,
        "y_pred": y_pred,
        "label": label,
        "protected_attr": groups
    })
    return df

def template_csv() -> bytes:
    cols = ["feature_x1","feature_x2","score","y_pred","label","protected_attr"]
    df = pd.DataFrame(columns=cols)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
