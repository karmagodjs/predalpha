# Track A data audit

Research dataset: 94 rows, columns ['timestamp', 'mid_price_up', 'spread_up', 'spread_down', 'mid_price_change', 'label_5m']. Missing values: {'timestamp': 0, 'mid_price_up': 0, 'spread_up': 0, 'spread_down': 0, 'mid_price_change': 0, 'label_5m': 0}. Duplicate timestamps: 0; ascending timestamps: True.

Labels: {'DOWN': 60, 'UP': 34}. The research rows exactly match the non-null-feature rows of `btc_oct3_aligned_5m_dataset.parquet`: True. The event source has 95 rows; 1 row was excluded because `mid_price_change` was missing. Event-source features/timestamps match research: True, but 7 labels differ because its target anchor differs.

## Target construction

The research set was reproduced from `build_oct3_aligned_5m_dataset.py`: each event is mapped to the latest fully closed BTC 1-minute candle; target is that candle close time + 5 minutes; `label_5m` is UP if `target_close - close > 0`, DOWN if negative, else FLAT. Future candle information is target-only, but the prediction timestamp can occur up to one minute after the anchor candle close, so the event-to-target horizon is not consistently five minutes. `build_oct3_event_5m_dataset.py` instead targets the first candle closed after event timestamp + five minutes, producing 7 different labels; it is not the source of the existing research labels.

## Split audit

- train: 65 rows, labels {'DOWN': 56, 'UP': 9}, bounds ['2026-10-02 18:26:50.945000+00:00', '2026-10-02 18:45:32.280000+00:00']
- validation: 14 rows, labels {'UP': 13, 'DOWN': 1}, bounds ['2026-10-02 18:45:51.809000+00:00', '2026-10-02 18:49:20.754000+00:00']
- test: 15 rows, labels {'UP': 12, 'DOWN': 3}, bounds ['2026-10-02 18:49:24.891000+00:00', '2026-10-02 18:56:16.565000+00:00']

Splits are chronologically ordered and non-overlapping. Timestamp spacing is irregular; there is no explicit purge gap, so this small, autocorrelated sample is unsuitable for strong out-of-sample claims.
