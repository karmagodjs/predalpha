# Prospective Shadow Collector Health Report — Session `session_live_010`

> [!IMPORTANT]
> **RESEARCH-ONLY PROSPECTIVE TELEMETRY**: Zero real capital deployed; zero live orders sent to exchange.
> All hypothetical returns represent simulated paper executions under conservative bid/ask crossing and 5 bps taker fees.

---

## 1. Session Status: **ABORTED (HARD STOP TRIGGERED)**

| Telemetry Metric | Session Value | Quality Benchmark |
| :--- | :---: | :--- |
| **Protocol Version** | `1.2` | Protocol Amendment v1.2 |
| **Start Timestamp (UTC)** | `2026-10-10T20:01:06.338829+00:00` | Recorded |
| **End Timestamp (UTC)** | `2026-10-10T20:05:29.096355+00:00` | Recorded |
| **Session Status** | `ABORTED_HARD_STOP` | Operational certification |
| **Raw WebSocket Frames** | `49,962` | Active feed connectivity |
| **JSON Payloads Parsed** | `49,933` | Normal market data |
| **Heartbeat / Control Frames** | `0` | PING/PONG/OK handled gracefully |
| **Protocol Status / Rejections** | `0` | 0 expected (handled without crash) |
| **Malformed / Non-JSON Frames** | `0` | 0 expected (logged safely) |
| **Queue Overflows (Dropped)** | `0` | 0 required for strict continuity |
| **Peak Ingestion Queue Depth** | `313` | Capacity: 10,000 |
| **Reconnect Attempts** | `0` | Resubscription & snapshot rebuild |
| **Market Rollovers** | `1` | Feature engine cleanly reset |
| **Eligible Grid Steps (Quote Validation)** | `251` | Denominator for missing quote rate |
| **Unquotable / Crossed Quotes** | `1` | Numerator for missing quote rate |
| **Missing Quote Rate** | `0.40%` | Threshold: <= 25.0% (abort if > 25%) |
| **Valid Staleness Observations** | `250` | Timestamped quote observations |
| **Average Quote Staleness** | `67.28 ms` | Threshold: <= 2000.0 ms (abort if > 2000 ms) |
| **Uninitialized Book Steps** | `1` | Steps waiting for initial snapshots (tracked separately) |
| **1-Second Decisions Logged** | `232` | Grid synchronization |
| **Feature Warm-up Steps** | `18` | Initial sequence building |
| **Trades Triggered (EXECUTE)** | `41` | Evaluated under frozen regime gates |
| **Trades Suppressed by Gates** | `144` | Strict regime gating |
| **Stale Quote Events** | `1` | Quote age > 2000ms threshold |
| **Buffered Events Drained** | `49,883` | Causal point-in-time event alignment |
| **Future-Dated Frames Rejected** | `0` | 0 expected (rejected safely) |
| **Invalid Timestamp Frames Rejected** | `0` | 0 expected (rejected safely) |
| **Event Buffer Overflows** | `0` | Capacity: 20,000 |

---

## 2. Invariants & Safety Verification

- [x] **Zero Live Orders**: Process operates strictly as a read-only WebSocket subscriber.
- [x] **Model Weights Frozen**: SmallLSTM weights remained bitwise identical (SHA-256 verified).
- [x] **Locked Test Isolation**: `test_scaled.npz` was never accessed or opened.
- [x] **Append-Only Logging**: Every observation hashed with SHA-256 rolling digest.
- [x] **Observation Log File**: `session_live_010_observations.jsonl` (SHA-256: `08e6b5dc9430713a...`).

---

## 3. Hard Stopping Abort Trigger

- **Trigger Code**: `CONSECUTIVE_LOSSES_LIMIT (8 >= 8)`

- **Protocol Version**: `1.2`

- **Missing Quote Rate**: `0.40%` (Threshold: `25.0%`)

- **Average Quote Staleness**: `67.28 ms` (Threshold: `2000.0 ms`)

- **Enforcement**: New hypothetical trade entries halted immediately; pending in-flight positions drained before session exit.

- **Initial Unquotable Diagnostic**: Reasons=`MISSING_OR_NONPOSITIVE_ASK`, GridTS=`1791662608000`, BookTS=`1791662608000`, Bid=`0.99`, Ask=`None`

---

## 4. Unquotable Quote Diagnostics (First 10 Samples)

| Grid TS (ms) | Book Update (ms) | Bid | Ask | Failure Reasons | Timestamp (UTC) |
| :---: | :---: | :---: | :---: | :--- | :--- |
| `1791662608000` | `1791662608000` | `0.9900` | `None` | `MISSING_OR_NONPOSITIVE_ASK` | `2026-10-10T20:03:28.022678+00:00` |


