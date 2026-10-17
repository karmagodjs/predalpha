# Prospective Shadow Collector Health Report — Session `session_live_003`

> [!IMPORTANT]
> **RESEARCH-ONLY PROSPECTIVE TELEMETRY**: Zero real capital deployed; zero live orders sent to exchange.
> All hypothetical returns represent simulated paper executions under conservative bid/ask crossing and 5 bps taker fees.

---

## 1. Session Status: **DEGRADED**

| Telemetry Metric | Session Value | Quality Benchmark |
| :--- | :---: | :--- |
| **Start Timestamp (UTC)** | `2026-10-09T20:11:07.462087+00:00` | Recorded |
| **End Timestamp (UTC)** | `2026-10-09T20:16:08.635851+00:00` | Recorded |
| **Session Status** | `DEGRADED` | Operational certification |
| **Raw WebSocket Frames** | `63,469` | Active feed connectivity |
| **JSON Payloads Parsed** | `63,468` | Normal market data |
| **Heartbeat / Control Frames** | `0` | PING/PONG/OK handled gracefully |
| **Malformed / Non-JSON Frames** | `1` | 0 expected (logged safely) |
| **Queue Overflows (Dropped)** | `0` | 0 required for strict continuity |
| **Peak Ingestion Queue Depth** | `243` | Capacity: 10,000 |
| **Reconnect Attempts** | `0` | Resubscription & snapshot rebuild |
| **Market Rollovers** | `1` | Feature engine cleanly reset |
| **1-Second Decisions Logged** | `115` | Grid synchronization |
| **Trades Triggered (EXECUTE)** | `57` | Evaluated under frozen regime gates |
| **Trades Suppressed by Gates** | `49` | Strict regime gating |
| **Predictions Skipped (FLAT)** | `9` | Non-directional model output |
| **Stale Quote Events** | `0` | Quote age > 2000ms threshold |

---

## 2. Invariants & Safety Verification

- [x] **Zero Live Orders**: Process operates strictly as a read-only WebSocket subscriber.
- [x] **Model Weights Frozen**: SmallLSTM weights remained bitwise identical (SHA-256 verified).
- [x] **Locked Test Isolation**: `test_scaled.npz` was never accessed or opened.
- [x] **Append-Only Logging**: Every observation hashed with SHA-256 rolling digest.
- [x] **Observation Log File**: `session_live_003_observations.jsonl` (SHA-256: `1442fc5ff01795cb...`).

---

## 4. Malformed Message Samples (First 5)

| Timestamp UTC | Length | Sample Snippet | Error Detail |
| :--- | :---: | :--- | :--- |
| `2026-10-09T20:15:00.219360+00:00` | 17 | `'INVALID OPERATION'` | `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` |


