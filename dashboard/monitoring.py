from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

def load_risk_status():
    candidates = [
        PROCESSED / "risk_status.json",
        PROCESSED / "risk_report.json",
    ]

    for path in candidates:
        if path.exists():
            try:
                import json
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data, str(path.relative_to(ROOT))
            except Exception as exc:
                return {}, f"Read error: {exc}"

    return {}, "Risk status file not found"

def load_execution_events():
    candidates = [
        PROCESSED / "execution_events.parquet",
        PROCESSED / "fills.parquet",
    ]

    for path in candidates:
        if path.exists():
            try:
                return pd.read_parquet(path), str(path.relative_to(ROOT))
            except Exception as exc:
                return pd.DataFrame(), f"Read error: {exc}"

    return pd.DataFrame(), "Execution events not found"
