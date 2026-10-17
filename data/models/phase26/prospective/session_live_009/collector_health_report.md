# Prospective Shadow Collector Health Report — Session `session_live_009`

> [!IMPORTANT]
> **RESEARCH-ONLY PROSPECTIVE TELEMETRY**: Zero real capital deployed; zero live orders sent to exchange.
> All hypothetical returns represent simulated paper executions under conservative bid/ask crossing and 5 bps taker fees.

---

## 1. Session Status: **ABORTED (HARD STOP TRIGGERED)**

| Telemetry Metric | Session Value | Quality Benchmark |
| :--- | :---: | :--- |
| **Protocol Version** | `1.1` | Protocol Amendment v1.1 |
| **Start Timestamp (UTC)** | `2026-10-10T18:06:23.518527+00:00` | Recorded |
| **End Timestamp (UTC)** | `2026-10-10T18:07:19.063247+00:00` | Recorded |
| **Session Status** | `ABORTED_HARD_STOP` | Operational certification |
| **Raw WebSocket Frames** | `12,771` | Active feed connectivity |
| **JSON Payloads Parsed** | `12,767` | Normal market data |
| **Heartbeat / Control Frames** | `0` | PING/PONG/OK handled gracefully |
| **Protocol Status / Rejections** | `0` | 0 expected (handled without crash) |
| **Malformed / Non-JSON Frames** | `0` | 0 expected (logged safely) |
| **Queue Overflows (Dropped)** | `0` | 0 required for strict continuity |
| **Peak Ingestion Queue Depth** | `109` | Capacity: 10,000 |
| **Reconnect Attempts** | `0` | Resubscription & snapshot rebuild |
| **Market Rollovers** | `0` | Feature engine cleanly reset |
| **Eligible Grid Steps (Quote Validation)** | `45` | Denominator for missing quote rate |
| **Unquotable / Crossed Quotes** | `0` | Numerator for missing quote rate |
| **Missing Quote Rate** | `0.00%` | Threshold: <= 25.0% (abort if > 25%) |
| **Valid Staleness Observations** | `45` | Timestamped quote observations |
| **Average Quote Staleness** | `11.42 ms` | Threshold: <= 2000.0 ms (abort if > 2000 ms) |
| **Uninitialized Book Steps** | `0` | Steps waiting for initial snapshots (tracked separately) |
| **1-Second Decisions Logged** | `36` | Grid synchronization |
| **Feature Warm-up Steps** | `9` | Initial sequence building |
| **Trades Triggered (EXECUTE)** | `19` | Evaluated under frozen regime gates |
| **Trades Suppressed by Gates** | `14` | Strict regime gating |
| **Stale Quote Events** | `0` | Quote age > 2000ms threshold |
| **Buffered Events Drained** | `12,767` | Causal point-in-time event alignment |
| **Future-Dated Frames Rejected** | `0` | 0 expected (rejected safely) |
| **Invalid Timestamp Frames Rejected** | `0` | 0 expected (rejected safely) |
| **Event Buffer Overflows** | `0` | Capacity: 20,000 |

---

## 2. Invariants & Safety Verification

- [x] **Zero Live Orders**: Process operates strictly as a read-only WebSocket subscriber.
- [x] **Model Weights Frozen**: SmallLSTM weights remained bitwise identical (SHA-256 verified).
- [x] **Locked Test Isolation**: `test_scaled.npz` was never accessed or opened.
- [x] **Append-Only Logging**: Every observation hashed with SHA-256 rolling digest.
- [x] **Observation Log File**: `session_live_009_observations.jsonl` (SHA-256: `06fd248b43b0d536...`).

---

## 3. Hard Stopping Abort Trigger

- **Trigger Code**: `CONSECUTIVE_LOSSES_LIMIT (8 >= 8)`

- **Protocol Version**: `1.1`

- **Missing Quote Rate**: `0.00%` (Threshold: `25.0%`)

- **Average Quote Staleness**: `11.42 ms` (Threshold: `2000.0 ms`)

- **Enforcement**: New hypothetical trade entries halted immediately; pending in-flight positions drained before session exit.

