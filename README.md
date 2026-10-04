# PredAlpha

## Phase 2–3 data collection

Collectors are intentionally separate from research labeling and model training. They append immutable JSONL records and can be safely restarted: records with an identical logical record ID are skipped, never overwritten.

### Setup and commands (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# One synthetic Track A write; no network needed
python scripts/collect_track_a.py --dry-run
# Live BTC collection for 10 minutes, every 60 seconds
python scripts/collect_track_a.py --start-time 2026-10-04T12:00:00Z --end-time 2026-10-04T12:10:00Z --interval-seconds 60
# Create a separate cleaned parquet and quality status (no labels)
python scripts/collect_track_a.py --process-only
# Verify/capture metadata for specified condition IDs; add --max-messages 100 for live bounded websocket capture
python scripts/collect_track_b.py <condition-id> --settlement-only
python scripts/collect_track_b.py <condition-id> --max-messages 100
# Discover candidate active BTC Up/Down markets (inspect candidates before capture)
python scripts/collect_track_b.py --discover
python scripts/report_collection_status.py
python -m pytest -q --basetemp .test-tmp
```

Track A raw schema (`data/raw/track_a/btc_observations.jsonl`): `source`, `symbol`, source/collection timestamps, OHLCV, raw source payload, and stable `record_id`. Clean output is `data/processed/track_a/btc_observations.parquet` and contains no future-dependent label.

Track B has one folder per condition ID, containing append-only `official_metadata.jsonl`, `events.jsonl`, `settlement_evidence.jsonl`, and `collection_errors.jsonl`. Token outcomes are accepted only from official CLOB metadata; a settlement is `VERIFIED` only when exactly one official token has `winner: true`. Repeated snapshots are correlated observations from one market, never independent evaluation samples.

Limitations: public endpoints and websocket access can rate-limit or disconnect; collection error logs and quality reports expose gaps rather than inventing observations. Discover results are candidates, not an assertion that they are five-minute markets—inspect official metadata before long captures. Raw files and collection parquet outputs are Git-ignored.
