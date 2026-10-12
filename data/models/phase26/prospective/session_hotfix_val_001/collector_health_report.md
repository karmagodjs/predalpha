# Prospective Shadow Collector Health Report — Session `session_hotfix_val_001`

> [!IMPORTANT]
> **RESEARCH-ONLY PROSPECTIVE TELEMETRY**: Zero real capital deployed; zero live orders sent to exchange.
> All hypothetical returns represent simulated paper executions under conservative bid/ask crossing and 5 bps taker fees.

---

## 1. Session Status: **FAILED**

| Telemetry Metric | Session Value | Quality Benchmark |
| :--- | :---: | :--- |
| **Start Timestamp (UTC)** | `2026-10-09T19:38:34.148443+00:00` | Recorded |
| **End Timestamp (UTC)** | `2026-10-09T19:38:35.589440+00:00` | Recorded |
| **Session Status** | `FAILED` | Operational certification |
| **Raw WebSocket Frames** | `0` | Active feed connectivity |
| **JSON Payloads Parsed** | `0` | Normal market data |
| **Heartbeat / Control Frames** | `0` | PING/PONG/OK handled gracefully |
| **Malformed / Non-JSON Frames** | `0` | 0 expected (logged safely) |
| **Queue Overflows (Dropped)** | `0` | 0 required for strict continuity |
| **Peak Ingestion Queue Depth** | `0` | Capacity: 10,000 |
| **Reconnect Attempts** | `0` | Resubscription & snapshot rebuild |
| **Market Rollovers** | `0` | Feature engine cleanly reset |
| **1-Second Decisions Logged** | `0` | Grid synchronization |
| **Trades Triggered (EXECUTE)** | `0` | Evaluated under frozen regime gates |
| **Trades Suppressed by Gates** | `0` | Strict regime gating |
| **Predictions Skipped (FLAT)** | `0` | Non-directional model output |
| **Stale Quote Events** | `0` | Quote age > 2000ms threshold |

---

## 2. Invariants & Safety Verification

- [x] **Zero Live Orders**: Process operates strictly as a read-only WebSocket subscriber.
- [x] **Model Weights Frozen**: SmallLSTM weights remained bitwise identical (SHA-256 verified).
- [x] **Locked Test Isolation**: `test_scaled.npz` was never accessed or opened.
- [x] **Append-Only Logging**: Every observation hashed with SHA-256 rolling digest.
- [x] **Observation Log File**: `session_hotfix_val_001_observations.jsonl` (SHA-256: `...`).

---

## 3. Fatal Error Diagnostics

```text
'ClientConnection' object has no attribute 'closed'
```

