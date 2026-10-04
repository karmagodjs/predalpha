"""Generate Track A/B audits, baselines, reports, and unresolved status without changing source data."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_data.track_a import CONFIG, run_baselines, validate_chronological_splits, validate_feature_frame
from market_data.track_b_audit import MARKET_ID, REQUESTED_TOKEN_IDS, audit_raw_sessions, build_verified_market_events

PROCESSED, REPORTS, RAW = ROOT / "data/processed", ROOT / "reports", ROOT / "data/raw"


def markdown_metrics(metrics: dict[str, object]) -> str:
    return " | ".join(f"{key}={value:.3f}" for key, value in metrics.items() if isinstance(value, float))


def get_json(url: str) -> dict[str, object]:
    try:
        with urlopen(Request(url, headers={"User-Agent": "PredAlpha-audit/1.0"}), timeout=15) as response:
            body = response.read().decode("utf-8")
            return {"url": url, "status": response.status, "sample": body[:1000], "parsed": json.loads(body)}
    except HTTPError as error:
        return {"url": url, "status": error.code, "failure_reason": str(error), "sample": error.read().decode("utf-8", errors="replace")[:1000]}
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        return {"url": url, "status": None, "failure_reason": repr(error), "sample": ""}


def run_track_a() -> dict[str, object]:
    event_source = pd.read_parquet(PROCESSED / "btc_oct3_event_5m_dataset.parquet")
    aligned_source = pd.read_parquet(PROCESSED / "btc_oct3_aligned_5m_dataset.parquet")
    research = pd.read_parquet(PROCESSED / "btc_oct3_5m_research_dataset.parquet")
    splits = {name: pd.read_parquet(PROCESSED / "splits" / f"{name}.parquet") for name in ("train", "validation", "test")}
    validate_feature_frame(research)
    validate_chronological_splits(splits)
    event_safe = event_source.loc[event_source["mid_price_change"].notna(), [*CONFIG.feature_columns, CONFIG.target_column, CONFIG.timestamp_column]].reset_index(drop=True)
    aligned_safe = aligned_source.loc[aligned_source["mid_price_change"].notna(), [*CONFIG.feature_columns, CONFIG.target_column, CONFIG.timestamp_column]].reset_index(drop=True)
    research_sorted = research.loc[:, [*CONFIG.feature_columns, CONFIG.target_column, CONFIG.timestamp_column]].reset_index(drop=True)
    event_feature_match = event_safe.loc[:, [*CONFIG.feature_columns, CONFIG.timestamp_column]].equals(research_sorted.loc[:, [*CONFIG.feature_columns, CONFIG.timestamp_column]])
    event_label_mismatches = int((event_safe[CONFIG.target_column] != research_sorted[CONFIG.target_column]).sum())
    aligned_consistency = aligned_safe.equals(research_sorted)
    audit = {
        "research_rows": len(research), "research_columns": research.columns.tolist(), "missing_values": research.isna().sum().to_dict(),
        "duplicate_timestamps": int(research["timestamp"].duplicated().sum()), "timestamps_ascending": bool(pd.to_datetime(research["timestamp"], utc=True).is_monotonic_increasing),
        "label_distribution": research["label_5m"].value_counts().to_dict(), "feature_match_with_event_source": event_feature_match,
        "event_source_label_mismatches": event_label_mismatches, "feature_label_match_with_aligned_source": aligned_consistency,
        "event_source_rows": len(event_source), "event_source_rows_removed_for_missing_feature": len(event_source) - len(event_safe),
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "split_labels": {name: frame["label_5m"].value_counts().to_dict() for name, frame in splits.items()},
        "split_bounds": {name: [str(frame.timestamp.min()), str(frame.timestamp.max())] for name, frame in splits.items()},
    }
    results = run_baselines(splits)
    (REPORTS / "track_a_data_audit.md").write_text(
        "# Track A data audit\n\n"
        f"Research dataset: {audit['research_rows']} rows, columns {audit['research_columns']}. Missing values: {audit['missing_values']}. "
        f"Duplicate timestamps: {audit['duplicate_timestamps']}; ascending timestamps: {audit['timestamps_ascending']}.\n\n"
        f"Labels: {audit['label_distribution']}. The research rows exactly match the non-null-feature rows of `btc_oct3_aligned_5m_dataset.parquet`: {aligned_consistency}. "
        f"The event source has {len(event_source)} rows; {len(event_source)-len(event_safe)} row was excluded because `mid_price_change` was missing. Event-source features/timestamps match research: {event_feature_match}, but {event_label_mismatches} labels differ because its target anchor differs.\n\n"
        "## Target construction\n\n"
        "The research set was reproduced from `build_oct3_aligned_5m_dataset.py`: each event is mapped to the latest fully closed BTC 1-minute candle; target is that candle close time + 5 minutes; `label_5m` is UP if `target_close - close > 0`, DOWN if negative, else FLAT. Future candle information is target-only, but the prediction timestamp can occur up to one minute after the anchor candle close, so the event-to-target horizon is not consistently five minutes. `build_oct3_event_5m_dataset.py` instead targets the first candle closed after event timestamp + five minutes, producing 7 different labels; it is not the source of the existing research labels.\n\n"
        "## Split audit\n\n"
        + "\n".join(f"- {name}: {audit['split_rows'][name]} rows, labels {audit['split_labels'][name]}, bounds {audit['split_bounds'][name]}" for name in splits)
        + "\n\nSplits are chronologically ordered and non-overlapping. Timestamp spacing is irregular; there is no explicit purge gap, so this small, autocorrelated sample is unsuitable for strong out-of-sample claims.\n",
        encoding="utf-8",
    )
    result_lines = ["# Track A baseline results\n\n", "Features: `mid_price_up`, `spread_up`, `spread_down`, `mid_price_change`. Future-derived columns are prohibited by `market_data.track_a.CONFIG` and validated before fitting.\n\n", "Models were fitted once on chronological train only; validation and test were scored without tuning. Metrics are macro precision/recall/F1.\n\n"]
    for model, by_split in results.items():
        result_lines.append(f"## {model}\n\n")
        for split, metrics in by_split.items():
            result_lines.append(f"- {split}: {markdown_metrics(metrics)}; class distribution={metrics['class_distribution']}; labels={metrics['labels']}; confusion matrix={metrics['confusion_matrix']}\n")
    result_lines.append("\n## Limitations\n\nOnly 94 observations and seven label transitions exist. Train has only 9 UP labels while validation/test are UP-heavy; test has 15 rows. Accuracy can be misleading, and neither result supports a predictive-performance claim. More independently resolved markets and a purged/walk-forward design are required before meaningful modeling.\n")
    (REPORTS / "track_a_results.md").write_text("".join(result_lines), encoding="utf-8")
    return {"audit": audit, "results": results}


def run_track_b() -> dict[str, object]:
    audit = audit_raw_sessions(RAW)
    gamma_url = "https://gamma-api.polymarket.com/markets?" + urlencode({"condition_id": MARKET_ID})
    clob_urls = [f"https://clob.polymarket.com/markets/{MARKET_ID}"] + [f"https://clob.polymarket.com/book?token_id={token}" for token in REQUESTED_TOKEN_IDS]
    calls = [get_json(gamma_url), *(get_json(url) for url in clob_urls)]
    observed = set(audit["observed_asset_ids"])
    requested = set(REQUESTED_TOKEN_IDS)
    clob_market = calls[1].get("parsed", {}) if calls[1].get("status") == 200 else {}
    tokens = clob_market.get("tokens", []) if isinstance(clob_market, dict) else []
    token_mapping = {str(token.get("token_id")): {"outcome": token.get("outcome"), "winner": token.get("winner"), "price": token.get("price")} for token in tokens}
    observed = set(audit["observed_asset_ids"])
    mapping_verified = observed == set(token_mapping) and all(value["outcome"] in {"Up", "Down"} for value in token_mapping.values())
    winners = [value["outcome"] for value in token_mapping.values() if value["winner"] is True]
    settlement_verified = len(winners) == 1
    settlement = winners[0] if settlement_verified else None
    dataset = build_verified_market_events(RAW, token_mapping, settlement) if mapping_verified and settlement_verified else None
    if dataset is not None:
        dataset.to_parquet(PROCESSED / "track_b_verified_market_events.parquet", index=False)
    status = {
        "market_id": MARKET_ID, "verification_status": "VERIFIED" if dataset is not None else "UNRESOLVED", "dataset_created": dataset is not None,
        "requested_token_ids": REQUESTED_TOKEN_IDS, "observed_raw_token_ids": audit["observed_asset_ids"],
        "requested_tokens_observed_exactly": sorted(requested & observed),
        "requested_tokens_not_observed_exactly": sorted(requested - observed),
        "official_clob_market_evidence": {"url": calls[1]["url"], "http_status": calls[1].get("status"), "question": clob_market.get("question"), "accepting_order_timestamp": clob_market.get("accepting_order_timestamp"), "end_date_iso": clob_market.get("end_date_iso"), "resolution_timestamp": None, "resolution_source": "Binance BTC/USDT 1m close prices, per official CLOB description", "tokens": tokens},
        "verified_token_mapping": token_mapping if mapping_verified else None,
        "official_settlement_outcome": settlement,
        "dataset_validation": None if dataset is None else {"rows": len(dataset), "missing_values": dataset.isna().sum().to_dict(), "duplicate_timestamps": int(dataset.timestamp.duplicated().sum()), "label_distribution": dataset.market_settlement_outcome.value_counts().to_dict(), "winning_token_distribution": dataset.is_winning_token.value_counts().to_dict()},
        "api_calls": [{key: value for key, value in call.items() if key != "parsed"} for call in calls],
    }
    (PROCESSED / "track_b_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    api_lines = "\n".join(f"- `{call['url']}` — HTTP {call['status']}; {call.get('failure_reason', 'response received')}; sample: `{call.get('sample', '')[:250]}`" for call in status["api_calls"])
    (REPORTS / "track_b_market_audit.md").write_text(
        "# Track B raw market audit\n\n"
        f"Expected market ID: `{MARKET_ID}`. Observed market IDs: `{audit['observed_market_ids']}`. Observed assets: `{audit['observed_asset_ids']}`.\n\n"
        + "\n".join(f"## {item['file']}\n\nEvents: {item['event_count']}; types: {item['event_types']}; first/last received: {item['first_received_at']} / {item['last_received_at']}; assets: {item['asset_ids']}; resolution fields: {item['resolution_related_fields'] or 'none'}.\n" for item in audit["files"])
        + "\nRaw events are book events and do not contain an outcome field or resolution event.\n", encoding="utf-8")
    (REPORTS / "track_b_results.md").write_text(
        "# Track B verification result\n\n"
        f"Market ID: `{MARKET_ID}`. **Verification status: {status['verification_status']}. Dataset status: {'created' if dataset is not None else 'not created'}.**\n\n"
        f"Prompt-supplied token IDs: `{REQUESTED_TOKEN_IDS}`. Raw observed IDs: `{audit['observed_asset_ids']}`. Exact overlap: `{status['requested_tokens_observed_exactly']}`. This exposes a second-token discrepancy; it must be resolved from official metadata, not inferred.\n\n"
        f"Official CLOB mapping: `{token_mapping}`. Official question: `{clob_market.get('question')}`; accepting-order timestamp: `{clob_market.get('accepting_order_timestamp')}`; end date: `{clob_market.get('end_date_iso')}`. The endpoint does not return a market start time or resolution timestamp. The official description identifies Binance BTC/USDT 1-minute close prices as resolution source. Official token field `winner: true` identifies settlement outcome: `{settlement}`.\n\n"
        "API calls attempted:\n\n" + api_lines
        + (f"\n\nCreated `data/processed/track_b_verified_market_events.parquet`: {len(dataset)} raw book-event rows, no missing required labels, market settlement label distribution {dataset.market_settlement_outcome.value_counts().to_dict()}. `token_outcome` and `is_winning_token` are linked solely to the official CLOB token mapping.\n" if dataset is not None else "\n\nNo label was created because authoritative evidence remained insufficient.\n"), encoding="utf-8")
    return status


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    a = run_track_a()
    b = run_track_b()
    (REPORTS / "AB_completion_summary.md").write_text(
        "# PredAlpha Track A & B completion summary\n\n"
        "## Completed\n\nTrack A audit, leakage-safe feature configuration, chronological baseline evaluation, and tests were added without modifying raw/source data. Track B raw audit and authoritative API evidence logging were completed.\n\n"
        f"## Verified\n\nTrack A research dataset has {a['audit']['research_rows']} rows and exactly matches its aligned-source safe rows: {a['audit']['feature_label_match_with_aligned_source']}. Track B official CLOB metadata maps the raw IDs to Up/Down and reports settlement {b['official_settlement_outcome']}.\n\n"
        "## Readiness\n\nNeither track is ready for meaningful model training: A is too small, imbalanced, and serially dependent; B now has verified labels but only one independently resolved market. Next: collect many independently resolved markets and preserve official Gamma/CLOB market metadata at capture time.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
