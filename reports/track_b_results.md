# Track B verification result

Market ID: `0x0b1e1b2f032a7668cc96455e75939666e25eceab070d4b9162da5a9da7784b68`. **Verification status: VERIFIED. Dataset status: created.**

Prompt-supplied token IDs: `['32664731353322490016663940157951165038326534411865610104571929420891932752044', '83368144530639073518031297526937851295808514385655078316894643930648829402']`. Raw observed IDs: `['32664731353322490016663940157951165038326534411865610104571929420891932752044', '83368144530639073518031297526937851295808514385655078316894643930693648829402']`. Exact overlap: `['32664731353322490016663940157951165038326534411865610104571929420891932752044']`. This exposes a second-token discrepancy; it must be resolved from official metadata, not inferred.

Official CLOB mapping: `{'32664731353322490016663940157951165038326534411865610104571929420891932752044': {'outcome': 'Up', 'winner': False, 'price': 0}, '83368144530639073518031297526937851295808514385655078316894643930693648829402': {'outcome': 'Down', 'winner': True, 'price': 1}}`. Official question: `Bitcoin Up or Down on October 3?`; accepting-order timestamp: `2026-10-01T16:07:08Z`; end date: `2026-10-03T00:00:00Z`. The endpoint does not return a market start time or resolution timestamp. The official description identifies Binance BTC/USDT 1-minute close prices as resolution source. Official token field `winner: true` identifies settlement outcome: `Down`.

API calls attempted:

- `https://gamma-api.polymarket.com/markets?condition_id=0x0b1e1b2f032a7668cc96455e75939666e25eceab070d4b9162da5a9da7784b68` — HTTP 200; response received; sample: `[{"id":"559651","question":"Xi Jinping out before 2027?","conditionId":"0xa467b14d51f01b957109d9cbb1d6c124fab2a089d52ed8f471d23c2812e743b7","slug":"xi-jinping-out-before-2027","resolutionSource":"","endDate":"2027-01-01T04:59:00Z","liquidity":"562838`
- `https://clob.polymarket.com/markets/0x0b1e1b2f032a7668cc96455e75939666e25eceab070d4b9162da5a9da7784b68` — HTTP 200; response received; sample: `{"enable_order_book":false,"active":true,"closed":true,"archived":false,"accepting_orders":false,"accepting_order_timestamp":"2026-10-01T16:07:08Z","minimum_order_size":5,"minimum_tick_size":0.001,"condition_id":"0x0b1e1b2f032a7668cc96455e75939666e25`
- `https://clob.polymarket.com/book?token_id=32664731353322490016663940157951165038326534411865610104571929420891932752044` — HTTP 404; HTTP Error 404: Not Found; sample: `{"error":"No orderbook exists for the requested token id"}
`
- `https://clob.polymarket.com/book?token_id=83368144530639073518031297526937851295808514385655078316894643930648829402` — HTTP 404; HTTP Error 404: Not Found; sample: `{"error":"No orderbook exists for the requested token id"}
`

Created `data/processed/track_b_verified_market_events.parquet`: 495 raw book-event rows, no missing required labels, market settlement label distribution {'Down': 495}. `token_outcome` and `is_winning_token` are linked solely to the official CLOB token mapping.
