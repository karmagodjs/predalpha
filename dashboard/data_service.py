from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

DATA_CANDIDATES = [
    "feature_market_data.parquet",
    "market_data.parquet",
]

def load_market_data():
    for filename in DATA_CANDIDATES:
        path = PROCESSED / filename
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if not df.empty:
                    return df, str(path.relative_to(ROOT))
            except Exception as exc:
                return pd.DataFrame(), f"Could not read {filename}: {exc}"
    return pd.DataFrame(), "No recognized market-data parquet found"

def load_optional_parquet(filenames):
    for filename in filenames:
        path = PROCESSED / filename
        if path.exists():
            try:
                return pd.read_parquet(path), str(path.relative_to(ROOT))
            except Exception as exc:
                return pd.DataFrame(), f"Could not read {filename}: {exc}"
    return pd.DataFrame(), "Not available"

def load_optional_csv(filenames):
    for filename in filenames:
        path = PROCESSED / filename
        if path.exists():
            try:
                return pd.read_csv(path), str(path.relative_to(ROOT))
            except Exception as exc:
                return pd.DataFrame(), f"Could not read {filename}: {exc}"
    return pd.DataFrame(), "Not available"

def latest_row(df):
    if df.empty:
        return None
    if "timestamp" in df.columns:
        try:
            df = df.sort_values("timestamp")
        except Exception:
            pass
    return df.iloc[-1]

def numeric_value(row, candidates):
    if row is None:
        return None
    for key in candidates:
        if key in row.index and pd.notna(row[key]):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return None
from dashboard.logging_config import get_logger

logger = get_logger("predalpha.data")
