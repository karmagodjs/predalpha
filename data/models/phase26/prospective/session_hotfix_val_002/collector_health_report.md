# Prospective Shadow Collector Health Report — Session `session_hotfix_val_002`

> [!IMPORTANT]
> **RESEARCH-ONLY PROSPECTIVE TELEMETRY**: Zero real capital deployed; zero live orders sent to exchange.
> All hypothetical returns represent simulated paper executions under conservative bid/ask crossing and 5 bps taker fees.

---

## 1. Session Status: **HEALTHY (COMPLETE)**

| Telemetry Metric | Session Value | Quality Benchmark |
| :--- | :---: | :--- |
| **Start Timestamp (UTC)** | `2026-10-09T20:05:11.470179+00:00` | Recorded |
| **End Timestamp (UTC)** | `2026-10-09T20:06:21.512156+00:00` | Recorded |
| **Session Status** | `COMPLETE` | Operational certification |
| **Raw WebSocket Frames** | `33,961` | Active feed connectivity |
| **JSON Payloads Parsed** | `33,960` | Normal market data |
| **Heartbeat / Control Frames** | `0` | PING/PONG/OK handled gracefully |
| **Malformed / Non-JSON Frames** | `0` | 0 expected (logged safely) |
| **Queue Overflows (Dropped)** | `0` | 0 required for strict continuity |
| **Peak Ingestion Queue Depth** | `225` | Capacity: 10,000 |
| **Reconnect Attempts** | `0` | Resubscription & snapshot rebuild |
| **Market Rollovers** | `0` | Feature engine cleanly reset |
| **1-Second Decisions Logged** | `50` | Grid synchronization |
| **Trades Triggered (EXECUTE)** | `12` | Evaluated under frozen regime gates |
| **Trades Suppressed by Gates** | `38` | Strict regime gating |
| **Predictions Skipped (FLAT)** | `0` | Non-directional model output |
| **Stale Quote Events** | `0` | Quote age > 2000ms threshold |

---

## 2. Invariants & Safety Verification

- [x] **Zero Live Orders**: Process operates strictly as a read-only WebSocket subscriber.
- [x] **Model Weights Frozen**: SmallLSTM weights remained bitwise identical (SHA-256 verified).
- [x] **Locked Test Isolation**: `test_scaled.npz` was never accessed or opened.
- [x] **Append-Only Logging**: Every observation hashed with SHA-256 rolling digest.
- [x] **Observation Log File**: `session_hotfix_val_002_observations.jsonl` (SHA-256: `ca26d13e3a3b1a94...`).

