# Prospective Shadow Collector Health Report — Session `session_live_004_e2e_verify`

> [!IMPORTANT]
> **RESEARCH-ONLY PROSPECTIVE TELEMETRY**: Zero real capital deployed; zero live orders sent to exchange.
> All hypothetical returns represent simulated paper executions under conservative bid/ask crossing and 5 bps taker fees.

---

## 1. Session Status: **HEALTHY (COMPLETE)**

| Telemetry Metric | Session Value | Quality Benchmark |
| :--- | :---: | :--- |
| **Start Timestamp (UTC)** | `2026-10-10T04:19:25.562297+00:00` | Recorded |
| **End Timestamp (UTC)** | `2026-10-10T04:23:35.613279+00:00` | Recorded |
| **Session Status** | `COMPLETE` | Operational certification |
| **Raw WebSocket Frames** | `94,751` | Active feed connectivity |
| **JSON Payloads Parsed** | `94,731` | Normal market data |
| **Heartbeat / Control Frames** | `0` | PING/PONG/OK handled gracefully |
| **Protocol Status / Rejections** | `0` | 0 expected (handled without crash) |
| **Malformed / Non-JSON Frames** | `0` | 0 expected (logged safely) |
| **Queue Overflows (Dropped)** | `0` | 0 required for strict continuity |
| **Peak Ingestion Queue Depth** | `495` | Capacity: 10,000 |
| **Reconnect Attempts** | `0` | Resubscription & snapshot rebuild |
| **Market Rollovers** | `1` | Feature engine cleanly reset |
| **1-Second Decisions Logged** | `196` | Grid synchronization |
| **Feature Warm-up Steps** | `9` | Initial sequence building |
| **Unquotable / Crossed Quotes** | `33` | 0 valid opportunities during unquotable book |
| **Uninitialized Book Steps** | `1` | Steps waiting for initial snapshots |
| **Trades Triggered (EXECUTE)** | `56` | Evaluated under frozen regime gates |
| **Trades Suppressed by Gates** | `140` | Strict regime gating |
| **Predictions Skipped (FLAT)** | `0` | Non-directional model output |
| **Stale Quote Events** | `0` | Quote age > 2000ms threshold |

---

## 2. Invariants & Safety Verification

- [x] **Zero Live Orders**: Process operates strictly as a read-only WebSocket subscriber.
- [x] **Model Weights Frozen**: SmallLSTM weights remained bitwise identical (SHA-256 verified).
- [x] **Locked Test Isolation**: `test_scaled.npz` was never accessed or opened.
- [x] **Append-Only Logging**: Every observation hashed with SHA-256 rolling digest.
- [x] **Observation Log File**: `session_live_004_e2e_verify_observations.jsonl` (SHA-256: `a62b6e01e6e55879...`).

