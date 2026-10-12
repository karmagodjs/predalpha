"""
Phase 27 Prospective Shadow Collector — Passive Maker Entry & Dual-Cohort Telemetry.

Connects to the Polymarket CLOB WebSocket and Gamma REST API to run genuine,
real-time, zero-capital prospective shadow execution under the signed-off Phase 27 Maker Protocol.

KEY ARCHITECTURAL & PROTOCOL INVARIANTS:
1. Read-Only Zero-Capital Execution: Strictly passive shadow telemetry; zero orders sent to Polymarket exchange.
2. Passive Maker Entry at Touch: UP contracts posted at Bid B_0, DOWN contracts posted at Ask A_0.
3. Strict Price-Through Matching:
   - BUY fills iff Ask <= B_0 or Bid < B_0. Touch only (Bid == B_0, Ask > B_0) is STRICTLY UNFILLED.
   - SELL fills iff Bid >= A_0 or Ask > A_0. Touch only (Ask == A_0, Bid < A_0) is STRICTLY UNFILLED.
4. 2,000 ms Entry Cancellation: Pending maker orders expire at 2,000 ms without taker fallback.
5. 5,000 ms Holding Horizon: Filled orders hold for exactly 5,000 ms (t_fill + 5000ms), exiting via
   market order at opposing touch.
6. Asymmetric Fee Schedule: 0 bps maker entry fee, 5 bps taker exit fee (0.0005 * P_exit).
7. Missing Exit Quote Handling: Position settled as INCOMPLETE_MISSING_EXIT with 1 spread penalty
   (-0.0100) + exit fee (retained in ITT and filled cohorts without survivorship bias).
8. Dual-Cohort Framework: Evaluates ITT Master Ledger (all eligible signals, unfilled = 0.0)
   separately from Filled-Trade Cohort.
9. Cluster-Robust Statistical Inference: Liang-Zeger cluster-robust SE, t-statistic, and
   Cameron-Miller empirical N_eff clustered by burst ID.
10. Strict Isolation & Freezing: Locked test dataset (test_scaled.npz) is NEVER accessed;
    Phase 19 checkpoint weights are verified bitwise invariant.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field, asdict
import datetime
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Union
import urllib.request
import uuid

import numpy as np
import torch
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_v2.models.phase26_prospective_collector import (
    LiveTokenOrderBook,
    OnlineCausalFeatureEngine,
    PointInTimeEventBuffer,
    BufferedMarketEvent,
    classify_and_parse_ws_message,
    get_current_5m_slot,
    fetch_market_tokens_for_slot,
    is_connection_closed,
    GAMMA_URL,
    WS_URL,
    SLOT_SECONDS,
    QUEUE_MAXSIZE,
    SCALER_PARAMS_PATH,
)
from pipeline_v2.models.phase27_maker_execution import (
    classify_l2_fill_event,
    check_price_through_fill,
    compute_cluster_robust_variance,
    compute_empirical_effective_sample_size,
    compute_dual_cohort_metrics,
    define_phase27_protocol,
    sha256_file,
    compute_hash_string,
    MakerOrder,
    TradeRecord,
    LOCKED_TEST_PATH,
    MODEL_PATH,
    FROZEN_MODEL_SHA256,
    PREDECLARED_CONFIDENCE_THRESHOLD,
    TRAIN_SPREAD_THRESHOLD,
    TRAIN_DEPTH_IMBALANCE_P90,
    ENTRY_TIMEOUT_MS,
    HOLDING_HORIZON_MS,
    MAX_QUOTE_AGE_MS,
    MAX_INTER_BURST_GAP_MS,
    MAKER_ENTRY_FEE_BPS,
    TAKER_EXIT_FEE_BPS,
    FORCED_LIQUIDATION_SPREAD_PENALTY,
    HURDLE_ITT_MEAN_PNL,
    HURDLE_FILLED_MEAN_PNL,
    HURDLE_PROFIT_FACTOR_FLOOR,
    TARGET_PROFIT_FACTOR,
    HURDLE_DECISIVE_WIN_RATE,
    HURDLE_OVERALL_WIN_RATE,
    TARGET_MIN_FILLED_TRADES,
    TARGET_MIN_ITT_SIGNALS,
    TARGET_MIN_BURSTS,
    TARGET_MIN_SESSIONS,
    MAX_DIRECTIONAL_ASYMMETRY,
    CLUSTER_ROBUST_T_CRITICAL,
    assign_session_scoped_clusters,
    aggregate_prospective_sessions,
)
from pipeline_v2.models.recurrent_models import SmallLSTMModule

logger = logging.getLogger("pipeline_v2.models.phase27_prospective")

PROSPECTIVE_BASE_DIR = ROOT / "data" / "models" / "phase27" / "prospective"
GLOBAL_ABORT_FILE = ROOT / "data" / "models" / "phase27" / "global_abort_ledger.json"
EMPTY_FILE_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# Circuit breaker thresholds
CONSECUTIVE_LOSSES_LIMIT = 8
MAX_DRAWDOWN_LIMIT = 2.50
MAX_MISSING_QUOTE_RATE = 0.25
MAX_AVERAGE_STALENESS_MS = 2000.0
GLOBAL_STRATEGY_ABORT_LIMIT = 3
ROLLING_ASYMMETRY_WINDOW = 50
ROLLING_ASYMMETRY_LIMIT = 0.40
TARGET_MIN_N_EFF = 40.0
DEFAULT_GAMMA_TIMEOUT_SECONDS = 5.0


async def fetch_market_tokens_for_slot_async(
    slot: int,
    timeout: float = DEFAULT_GAMMA_TIMEOUT_SECONDS,
) -> Tuple[str, Dict[str, str]]:
    """
    Asynchronously resolve 5-minute Bitcoin Up/Down market token IDs from Gamma REST API.
    Executes blocking HTTP socket queries and JSON decoding off the main event loop thread
    via asyncio.to_thread, guaranteeing that network delays and DNS lookups never block the asyncio loop.
    Enforces a strict timeout boundary via asyncio.wait_for.
    """
    return await asyncio.wait_for(
        asyncio.to_thread(fetch_market_tokens_for_slot, slot, timeout),
        timeout=timeout,
    )


# -----------------------------------------------------------------------------
# Section 1: Global Strategy Abort Ledger
# -----------------------------------------------------------------------------

class GlobalStrategyAbortLedger:
    """
    Manages global strategy abort persistence across multiple prospective sessions.
    Distinguishes between strategy-risk aborts (consecutive losses >= 8, max drawdown >= 2.50)
    and infrastructure failures (WebSocket reconnect exhaustion, degraded book, quote missing).
    """

    def __init__(self, ledger_path: Path = GLOBAL_ABORT_FILE):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.strategy_abort_count = int(data.get("strategy_abort_count", 0))
                self.infrastructure_invalidated_count = int(data.get("infrastructure_invalidated_count", 0))
                self.aborted_sessions = list(data.get("aborted_sessions", []))
            except Exception as e:
                logger.warning("Error reading global abort ledger (%s), resetting: %s", self.ledger_path, e)
                self.strategy_abort_count = 0
                self.infrastructure_invalidated_count = 0
                self.aborted_sessions = []
        else:
            self.strategy_abort_count = 0
            self.infrastructure_invalidated_count = 0
            self.aborted_sessions = []

    def _save(self) -> None:
        data = {
            "strategy_abort_count": self.strategy_abort_count,
            "strategy_abort_limit": GLOBAL_STRATEGY_ABORT_LIMIT,
            "infrastructure_invalidated_count": self.infrastructure_invalidated_count,
            "is_globally_aborted": bool(self.strategy_abort_count >= GLOBAL_STRATEGY_ABORT_LIMIT),
            "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "aborted_sessions": self.aborted_sessions,
        }
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def record_abort(
        self,
        session_id: str,
        abort_type: str,  # "STRATEGY_RISK" or "INFRASTRUCTURE_FAILURE"
        reason: str,
        timestamp_utc: Optional[str] = None,
    ) -> None:
        """Record a session abort event and increment appropriate counter."""
        ts = timestamp_utc or datetime.datetime.now(datetime.timezone.utc).isoformat()
        if abort_type == "STRATEGY_RISK":
            self.strategy_abort_count += 1
        else:
            self.infrastructure_invalidated_count += 1

        self.aborted_sessions.append({
            "session_id": session_id,
            "abort_type": abort_type,
            "reason": reason,
            "timestamp_utc": ts,
        })
        self._save()

    def is_globally_aborted(self) -> bool:
        """Check if global strategy abort limit has been reached."""
        return self.strategy_abort_count >= GLOBAL_STRATEGY_ABORT_LIMIT


# -----------------------------------------------------------------------------
# Section 2: Phase 27 Append-Only Cryptographic Shadow Logger
# -----------------------------------------------------------------------------

class Phase27ShadowLogger:
    """
    Append-only shadow execution logger with rolling SHA-256 cryptographic hash chain.
    Records every 1-second grid decision, passive limit entry, fill evaluation,
    cancellation, and holding horizon exit.
    """

    def __init__(
        self,
        output_dir: Path,
        run_id: str,
        max_quote_age_ms: float = MAX_QUOTE_AGE_MS,
        max_data_gap_ms: float = MAX_INTER_BURST_GAP_MS,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.log_file = self.output_dir / f"{self.run_id}_observations.jsonl"
        self.manifest_file = self.output_dir / f"{self.run_id}_manifest.json"

        # Session-artifact collision protection
        if self.log_file.exists():
            raise FileExistsError(f"Artifact collision: log file already exists at {self.log_file}")
        if self.manifest_file.exists():
            raise FileExistsError(f"Artifact collision: manifest file already exists at {self.manifest_file}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_quote_age_ms = max_quote_age_ms
        self.max_data_gap_ms = max_data_gap_ms

        self.record_count = 0
        self.rolling_hash = hashlib.sha256()
        self._is_closed = False

        # Touch log file immediately to ensure existence
        self.log_file.touch(exist_ok=False)

    def log_observation(
        self,
        event_timestamp_ms: int,
        market_id: str,
        asset_id: str,
        quote_timestamp_ms: int,
        bid_entry: float,
        ask_entry: float,
        depth_imbalance: float,
        scaled_spread: float,
        abs_scaled_depth_imbal: float,
        model_probs: Tuple[float, float, float],
        predicted_class: int,
        confidence: float,
        burst_id: int,
        data_gap_ms: float,
        gate_decision: str,  # "EXECUTE", "REJECT", "SKIP"
        rejection_reasons: List[str],
        signal_id: Optional[str] = None,
        order: Optional[MakerOrder] = None,
        trade_record: Optional[TradeRecord] = None,
        hard_stop_active: bool = False,
    ) -> Dict[str, Any]:
        """Record decision point, order state, and outcome to append-only jsonl log."""
        if self._is_closed:
            raise RuntimeError("Phase27ShadowLogger already closed.")

        self.record_count += 1
        quote_age_ms = float(event_timestamp_ms - quote_timestamp_ms)
        event_dt_utc = datetime.datetime.fromtimestamp(event_timestamp_ms / 1000.0, datetime.timezone.utc).isoformat()
        decision_dt_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        is_directional = predicted_class in (0, 2)

        record: Dict[str, Any] = {
            "run_id": self.run_id,
            "record_index": self.record_count,
            "signal_id": signal_id,
            "event_timestamp_ms": event_timestamp_ms,
            "event_timestamp_utc": event_dt_utc,
            "decision_timestamp_utc": decision_dt_utc,
            "market_id": market_id,
            "asset_id": asset_id,
            "recording_burst_id": burst_id,
            "quote_timestamp_ms": quote_timestamp_ms,
            "quote_age_ms": round(quote_age_ms, 2),
            "data_gap_ms": round(data_gap_ms, 2),
            "bid_entry": float(bid_entry),
            "ask_entry": float(ask_entry),
            "spread": round(float(ask_entry - bid_entry), 6),
            "depth_imbalance": round(float(depth_imbalance), 6),
            "scaled_spread": round(float(scaled_spread), 6),
            "abs_scaled_depth_imbal": round(float(abs_scaled_depth_imbal), 6),
            "prob_down": round(float(model_probs[0]), 6),
            "prob_flat": round(float(model_probs[1]), 6),
            "prob_up": round(float(model_probs[2]), 6),
            "predicted_class": int(predicted_class),
            "confidence": round(float(confidence), 6),
            "is_directional": bool(is_directional),
            "gate_decision": gate_decision,
            "rejection_reasons": rejection_reasons,
            "hard_stop_active": bool(hard_stop_active),
            "side": order.side if order else None,
            "limit_price": order.limit_price if order else None,
            "order_quantity": order.order_quantity if order else 0.0,
            "filled_quantity": trade_record.filled_quantity if trade_record else (order.filled_quantity if order else 0.0),
            "fill_rate_ratio": trade_record.fill_rate_ratio if trade_record else (float(order.filled_quantity / order.order_quantity) if order else 0.0),
            "fill_timestamp_ms": trade_record.fill_timestamp_ms if trade_record else (order.fill_timestamp_ms if order else None),
            "fill_price": trade_record.fill_price if trade_record else (order.fill_price if order else None),
            "exit_timestamp_ms": trade_record.exit_timestamp_ms if trade_record else None,
            "exit_price": trade_record.exit_price if trade_record else None,
            "gross_pnl": round(trade_record.gross_pnl, 6) if trade_record else 0.0,
            "maker_fee": round(trade_record.maker_fee, 6) if trade_record else 0.0,
            "taker_fee": round(trade_record.taker_fee, 6) if trade_record else 0.0,
            "net_pnl": round(trade_record.net_pnl, 6) if trade_record else 0.0,
            "outcome_status": trade_record.outcome_status if trade_record else (order.status if order else ("SKIPPED_FLAT" if predicted_class == 1 else "SUPPRESSED_GATE")),
            "is_filled": bool(trade_record.is_filled) if trade_record else False,
            "is_itt": bool(order is not None or (trade_record is not None and trade_record.is_itt)),
            "fill_certainty": (getattr(trade_record, "fill_certainty", None) if trade_record else (getattr(order, "fill_certainty", None) if order else None)),
        }

        # Calculate row integrity hash and update rolling chain
        record_json = json.dumps(record, sort_keys=True)
        record_hash = compute_hash_string(record_json)
        record["integrity_hash"] = record_hash
        self.rolling_hash.update(record_hash.encode("utf-8"))

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def close(self) -> Dict[str, Any]:
        """Finalize append-only log and return cryptographic integrity manifest."""
        if self._is_closed:
            raise RuntimeError("Phase27ShadowLogger already closed.")
        self._is_closed = True

        log_sha = sha256_file(self.log_file) if self.log_file.exists() else EMPTY_FILE_SHA256
        manifest: Dict[str, Any] = {
            "run_id": self.run_id,
            "total_records": self.record_count,
            "log_file": str(self.log_file.name),
            "log_sha256": log_sha,
            "rolling_chain_hash": self.rolling_hash.hexdigest(),
            "closed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest


# -----------------------------------------------------------------------------
# Section 3: Phase 27 Prospective Shadow Session Runner
# -----------------------------------------------------------------------------

class Phase27ProspectiveSession:
    """
    Manages a live prospective shadow collector session under the signed-off Phase 27 Maker Protocol:
    - Decoupled fast WebSocket reader task preventing slow-consumer disconnects (close code 1013).
    - Passive maker order placement at the touch (Bid for BUY/UP, Ask for SELL/DOWN).
    - Strict price-through maker fill evaluation (rejecting touch-only fills).
    - 2,000 ms entry timeout cancellation with zero taker fallback on entry.
    - 5,000 ms physical holding horizon with 5 bps taker exit fee.
    - Zero survivorship bias for missing exits, cancellations, partial fills, and overlapping positions.
    - Multi-session global strategy-abort tracking (halting after 3 strategy aborts).
    - Complete dual-cohort performance accounting (ITT Master Ledger and Filled-Trade Cohort).
    - Guaranteed try/finally finalization flushing pending and open orders on all exit paths.
    """

    def __init__(
        self,
        output_base_dir: Path = PROSPECTIVE_BASE_DIR,
        model_path: Path = MODEL_PATH,
        session_id: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        max_observations: Optional[int] = None,
        queue_maxsize: int = QUEUE_MAXSIZE,
        global_abort_file: Path = GLOBAL_ABORT_FILE,
        token_resolver: Optional[Callable[[int], Awaitable[Tuple[str, Dict[str, str]]]]] = None,
    ):
        self.token_resolver = token_resolver or fetch_market_tokens_for_slot_async
        # Strict locked test set isolation invariant
        if not LOCKED_TEST_PATH.exists():
            raise FileNotFoundError(f"Safety check failed: locked test path {LOCKED_TEST_PATH} not found.")

        self.output_base_dir = Path(output_base_dir)
        self.session_id = session_id or f"p27_shadow_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.session_dir = self.output_base_dir / self.session_id

        # Artifact collision protection
        if self.session_dir.exists():
            raise FileExistsError(f"Session directory already exists: {self.session_dir}")

        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=False)

        self.duration_seconds = duration_seconds
        self.max_observations = max_observations
        self.queue_maxsize = queue_maxsize

        # Model verification
        self.model_path = Path(model_path)
        self.model_initial_hash = sha256_file(self.model_path)
        if self.model_initial_hash != FROZEN_MODEL_SHA256:
            raise RuntimeError(f"Model checkpoint hash mismatch! Expected {FROZEN_MODEL_SHA256}, got {self.model_initial_hash}")

        # Load SmallLSTM for inference
        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.model = SmallLSTMModule(input_size=11, hidden_size=32, num_layers=1, num_classes=3)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        # Engine and logger components
        self.feature_engine = OnlineCausalFeatureEngine(scaler_params_path=SCALER_PARAMS_PATH)
        self.shadow_logger = Phase27ShadowLogger(output_dir=self.session_dir, run_id=self.session_id)
        self.global_abort_ledger = GlobalStrategyAbortLedger(ledger_path=global_abort_file)

        # Queues and buffers
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self.event_buffer = PointInTimeEventBuffer(maxsize=queue_maxsize * 2, future_tolerance_ms=5000)

        # Live Maker Order & Position Ledgers
        self.pending_orders: List[MakerOrder] = []
        self.open_positions: List[MakerOrder] = []
        self.trade_records: List[TradeRecord] = []
        self.signal_counter: int = 0

        # State flags
        self.is_running = False
        self.is_book_valid = False
        self.is_book_degraded = False
        self.is_hard_stopped = False
        self.hard_stop_reason: Optional[str] = None
        self.hard_stop_type: Optional[str] = None  # "STRATEGY_RISK" or "INFRASTRUCTURE_FAILURE"
        self._is_finalized = False

        # Session telemetry
        self.start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.ws_messages_received = 0
        self.json_messages_parsed = 0
        self.control_messages_count = 0
        self.protocol_errors_count = 0
        self.malformed_messages_count = 0
        self.queue_overflow_count = 0
        self.event_buffer_overflow_count = 0
        self.max_queue_depth_seen = 0
        self.reconnect_count = 0
        self.rollover_events_count = 0
        self.buffered_events_drained_count = 0

        self.observations_logged = 0
        self.signals_generated_count = 0
        self.maker_fills_count = 0
        self.maker_cancellations_count = 0
        self.trades_suppressed = 0
        self.trades_skipped_flat = 0
        self.warmup_steps_count = 0
        self.unquotable_quotes_count = 0
        self.uninitialized_book_steps_count = 0
        self.stale_quotes_count = 0
        self.eligible_grid_steps_count = 0
        self.valid_staleness_observation_count = 0
        self.total_quote_age_ms = 0.0

        self.burst_id = 0
        self.last_signal_timestamp_ms: Optional[int] = None
        self.last_event_ms = 0
        self.session_fatal_error: Optional[str] = None
        self.malformed_samples: List[Dict[str, Any]] = []
        self.unquotable_diagnostics: List[Dict[str, Any]] = []
        self.rollover_records: List[Dict[str, Any]] = []

        # Risk state
        self.consecutive_losses = 0
        self.cum_net_pnl = 0.0
        self.peak_net_pnl = 0.0
        self.max_drawdown_seen = 0.0

    @property
    def missing_quote_rate(self) -> float:
        if self.eligible_grid_steps_count == 0:
            return 0.0
        return float(self.unquotable_quotes_count / self.eligible_grid_steps_count)

    @property
    def average_quote_age_ms(self) -> float:
        if self.valid_staleness_observation_count == 0:
            return 0.0
        return float(self.total_quote_age_ms / self.valid_staleness_observation_count)

    def _check_hard_stops(self) -> None:
        """Evaluate intra-session hard-stopping circuit breakers."""
        if self.is_hard_stopped:
            return

        # 1. Missing quote rate threshold (> 25%)
        if self.eligible_grid_steps_count >= 10 and self.missing_quote_rate > MAX_MISSING_QUOTE_RATE:
            self.is_hard_stopped = True
            self.hard_stop_type = "INFRASTRUCTURE_FAILURE"
            self.hard_stop_reason = f"MISSING_QUOTE_RATE_LIMIT ({self.missing_quote_rate:.4f} > {MAX_MISSING_QUOTE_RATE:.2f})"
            logger.warning("Hard stop triggered: %s", self.hard_stop_reason)
            return

        # 2. Average quote staleness threshold (> 2000 ms)
        if self.valid_staleness_observation_count >= 10 and self.average_quote_age_ms > MAX_AVERAGE_STALENESS_MS:
            self.is_hard_stopped = True
            self.hard_stop_type = "INFRASTRUCTURE_FAILURE"
            self.hard_stop_reason = f"MAX_STALENESS_LIMIT ({self.average_quote_age_ms:.1f}ms > {MAX_AVERAGE_STALENESS_MS:.1f}ms)"
            logger.warning("Hard stop triggered: %s", self.hard_stop_reason)
            return

        # 3. Consecutive losses limit (>= 8)
        if self.consecutive_losses >= CONSECUTIVE_LOSSES_LIMIT:
            self.is_hard_stopped = True
            self.hard_stop_type = "STRATEGY_RISK"
            self.hard_stop_reason = f"CONSECUTIVE_LOSSES_LIMIT ({self.consecutive_losses} >= {CONSECUTIVE_LOSSES_LIMIT})"
            logger.warning("Strategy hard stop triggered: %s", self.hard_stop_reason)
            return

        # 4. Max drawdown limit (>= 2.50 units)
        if self.max_drawdown_seen >= MAX_DRAWDOWN_LIMIT:
            self.is_hard_stopped = True
            self.hard_stop_type = "STRATEGY_RISK"
            self.hard_stop_reason = f"MAX_DRAWDOWN_LIMIT ({self.max_drawdown_seen:.4f} >= {MAX_DRAWDOWN_LIMIT:.2f})"
            logger.warning("Strategy hard stop triggered: %s", self.hard_stop_reason)
            return

        # 5. Rolling directional asymmetry over last 50 filled trades (> 0.40)
        filled_trades = [r for r in self.trade_records if r.is_filled]
        if len(filled_trades) >= ROLLING_ASYMMETRY_WINDOW:
            recent_50 = filled_trades[-ROLLING_ASYMMETRY_WINDOW:]
            up_50 = sum(1 for r in recent_50 if r.predicted_class == 2)
            down_50 = sum(1 for r in recent_50 if r.predicted_class == 0)
            asym_50 = abs(up_50 - down_50) / float(ROLLING_ASYMMETRY_WINDOW)
            if asym_50 > ROLLING_ASYMMETRY_LIMIT:
                self.is_hard_stopped = True
                self.hard_stop_type = "STRATEGY_RISK"
                self.hard_stop_reason = f"ROLLING_DIRECTIONAL_ASYMMETRY_LIMIT ({asym_50:.4f} > {ROLLING_ASYMMETRY_LIMIT:.2f})"
                logger.warning("Strategy hard stop triggered: %s", self.hard_stop_reason)
                return

    async def _ws_reader_task(self, ws: Any) -> None:
        """Fast dedicated WebSocket reader task."""
        while self.is_running:
            try:
                msg = await ws.recv()
                recv_ms = int(time.time() * 1000)
                self.ws_messages_received += 1

                depth = self.message_queue.qsize()
                if depth > self.max_queue_depth_seen:
                    self.max_queue_depth_seen = depth

                try:
                    self.message_queue.put_nowait((msg, recv_ms))
                except asyncio.QueueFull:
                    self.queue_overflow_count += 1
                    self.is_book_degraded = True
                    logger.warning("WebSocket queue overflow (depth=%d). Book marked degraded.", depth)
            except asyncio.CancelledError:
                raise
            except ConnectionClosed as conn_err:
                logger.info("WebSocket connection closed in reader task: %s", conn_err)
                raise
            except (ConnectionError, OSError) as net_err:
                logger.warning("WebSocket network error in reader task: %s", net_err)
                raise
            except Exception as recv_err:
                logger.warning("ws_reader_task error: %s", recv_err)
                raise

    def _drain_message_queue_into_buffer(
        self,
        slug: str,
        up_book: LiveTokenOrderBook,
        down_book: LiveTokenOrderBook,
        max_drain: int = 500,
    ) -> int:
        """Classify and stage raw messages into PointInTimeEventBuffer."""
        drain_count = 0
        while not self.message_queue.empty() and drain_count < max_drain:
            drain_count += 1
            item = self.message_queue.get_nowait()
            raw_msg, recv_ms = item if isinstance(item, tuple) else (item, int(time.time() * 1000))

            category, parsed_data, err = classify_and_parse_ws_message(raw_msg)
            if category in ("EMPTY", "CONTROL"):
                self.control_messages_count += 1
            elif category == "PROTOCOL_ERROR":
                self.protocol_errors_count += 1
                logger.warning("Protocol error frame received (%s): %s", slug, raw_msg)
                if not (up_book.is_initialized and down_book.is_initialized):
                    raise ConnectionClosedOK(None, None)
            elif category == "MALFORMED":
                self.malformed_messages_count += 1
                if len(self.malformed_samples) < 50:
                    self.malformed_samples.append({
                        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "snippet": repr(raw_msg[:120]),
                        "error": err,
                    })
            elif category == "JSON":
                self.json_messages_parsed += 1
                pushed = self.event_buffer.push(
                    receipt_timestamp_ms=recv_ms,
                    data=parsed_data,
                    raw_message=raw_msg,
                )
                if not pushed:
                    self.event_buffer_overflow_count = self.event_buffer.overflow_count
                    self.is_book_degraded = True

        return drain_count

    def _apply_parsed_market_data(
        self,
        data: Any,
        up_book: LiveTokenOrderBook,
        down_book: LiveTokenOrderBook,
        now_ms: int,
        up_token_id: str,
        down_token_id: str,
    ) -> None:
        """Apply parsed events to live token order books."""
        events = data if isinstance(data, list) else [data]
        for ev in events:
            if not isinstance(ev, dict):
                continue
            ev_type = ev.get("event_type", ev.get("type"))
            aid = str(ev.get("asset_id", ""))

            if ev_type == "book":
                if aid == up_token_id:
                    up_book.apply_book_event(ev, now_ms)
                elif aid == down_token_id:
                    down_book.apply_book_event(ev, now_ms)
            elif ev_type == "price_change":
                for ch in ev.get("price_changes", []):
                    ch_aid = str(ch.get("asset_id", ""))
                    if ch_aid == up_token_id:
                        up_book.apply_price_change(ch, now_ms)
                    elif ch_aid == down_token_id:
                        down_book.apply_price_change(ch, now_ms)

    def _evaluate_pending_orders_and_open_positions(
        self,
        current_timestamp_ms: int,
        up_book: LiveTokenOrderBook,
        down_book: LiveTokenOrderBook,
        market_id: str,
    ) -> None:
        """
        Evaluate maker fills, 2000 ms timeout cancellations, and 5000 ms horizon exits.
        Called on every grid step with current order books.
        """
        valid_books = (
            up_book.is_initialized
            and down_book.is_initialized
            and not self.is_book_degraded
        )

        bid = up_book.best_bid
        ask = up_book.best_ask
        valid_quote = (
            valid_books
            and bid is not None
            and ask is not None
            and bid > 0
            and ask > 0
            and bid < ask
        )

        # ---------------------------------------------------------------------
        # 1. Settle Due Open Positions at Horizon (t_fill + 5000ms)
        # ---------------------------------------------------------------------
        for pos in list(self.open_positions):
            if current_timestamp_ms >= pos.target_exit_timestamp_ms:
                self.open_positions.remove(pos)

                target_book = down_book if pos.predicted_class == 0 or (getattr(down_book, "asset_id", "") and pos.asset_id == down_book.asset_id) else up_book
                target_bid = target_book.best_bid
                target_ask = target_book.best_ask
                valid_exit = (
                    valid_books
                    and target_bid is not None
                    and target_ask is not None
                    and target_bid > 0
                    and target_ask > 0
                    and target_bid < target_ask
                )

                if valid_exit:
                    exit_p = float(target_bid if pos.side == "BUY" else target_ask)
                    gross_pnl = float((exit_p - pos.fill_price) if pos.side == "BUY" else (pos.fill_price - exit_p)) * pos.filled_quantity
                    maker_fee = float(pos.fill_price * pos.filled_quantity * (MAKER_ENTRY_FEE_BPS / 10000.0))
                    taker_fee = float(exit_p * pos.filled_quantity * (TAKER_EXIT_FEE_BPS / 10000.0))
                    net_pnl = float(gross_pnl - maker_fee - taker_fee)
                    outcome = "COMPLETED"
                else:
                    exit_p = None
                    gross_pnl = float(FORCED_LIQUIDATION_SPREAD_PENALTY * pos.filled_quantity)
                    maker_fee = float(pos.fill_price * pos.filled_quantity * (MAKER_ENTRY_FEE_BPS / 10000.0))
                    taker_fee = float(pos.fill_price * pos.filled_quantity * (TAKER_EXIT_FEE_BPS / 10000.0))
                    net_pnl = float(gross_pnl - maker_fee - taker_fee)
                    outcome = "INCOMPLETE_MISSING_EXIT"

                record = TradeRecord(
                    signal_id=pos.signal_id,
                    record_index=len(self.trade_records),
                    decision_timestamp_ms=pos.decision_timestamp_ms,
                    market_id=pos.market_id,
                    asset_id=pos.asset_id,
                    burst_id=pos.burst_id,
                    side=pos.side,
                    predicted_class=pos.predicted_class,
                    confidence=pos.confidence,
                    scaled_spread=pos.scaled_spread,
                    abs_scaled_depth_imbal=pos.abs_scaled_depth_imbal,
                    limit_price=pos.limit_price,
                    order_quantity=pos.order_quantity,
                    filled_quantity=pos.filled_quantity,
                    fill_rate_ratio=float(pos.filled_quantity / pos.order_quantity),
                    fill_timestamp_ms=pos.fill_timestamp_ms,
                    fill_price=pos.fill_price,
                    exit_timestamp_ms=current_timestamp_ms,
                    exit_price=exit_p,
                    gross_pnl=gross_pnl,
                    maker_fee=maker_fee,
                    taker_fee=taker_fee,
                    net_pnl=net_pnl,
                    outcome_status=outcome,
                    is_filled=True,
                    is_itt=True,
                    fill_certainty=getattr(pos, "fill_certainty", None),
                    session_id=self.session_id,
                )
                self.trade_records.append(record)

                # Update risk state
                if net_pnl < -1e-6:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

                self.cum_net_pnl += net_pnl
                if self.cum_net_pnl > self.peak_net_pnl:
                    self.peak_net_pnl = self.cum_net_pnl
                dd = self.peak_net_pnl - self.cum_net_pnl
                if dd > self.max_drawdown_seen:
                    self.max_drawdown_seen = dd

                self._check_hard_stops()

        # ---------------------------------------------------------------------
        # 2. Evaluate Pending Maker Orders (Fill or 2000ms Timeout)
        # ---------------------------------------------------------------------
        for order in list(self.pending_orders):
            if current_timestamp_ms < order.decision_timestamp_ms:
                continue

            # Check timeout expiration
            if current_timestamp_ms > order.timeout_timestamp_ms:
                self.pending_orders.remove(order)
                order.status = "CANCELLED_UNFILLED"
                order.fill_certainty = "UNFILLED"
                self.maker_cancellations_count += 1

                # Record non-filled signal in ITT Master Ledger (PnL = 0.0)
                record = TradeRecord(
                    signal_id=order.signal_id,
                    record_index=len(self.trade_records),
                    decision_timestamp_ms=order.decision_timestamp_ms,
                    market_id=order.market_id,
                    asset_id=order.asset_id,
                    burst_id=order.burst_id,
                    side=order.side,
                    predicted_class=order.predicted_class,
                    confidence=order.confidence,
                    scaled_spread=order.scaled_spread,
                    abs_scaled_depth_imbal=order.abs_scaled_depth_imbal,
                    limit_price=order.limit_price,
                    order_quantity=order.order_quantity,
                    filled_quantity=0.0,
                    fill_rate_ratio=0.0,
                    fill_timestamp_ms=None,
                    fill_price=None,
                    exit_timestamp_ms=None,
                    exit_price=None,
                    gross_pnl=0.0,
                    maker_fee=0.0,
                    taker_fee=0.0,
                    net_pnl=0.0,
                    outcome_status="CANCELLED_UNFILLED",
                    is_filled=False,
                    is_itt=True,
                    fill_certainty="UNFILLED",
                    session_id=self.session_id,
                )
                self.trade_records.append(record)
                continue

            # Check strict price-through fill against order's specific target book
            target_book = down_book if order.predicted_class == 0 or (getattr(down_book, "asset_id", "") and order.asset_id == down_book.asset_id) else up_book
            t_bid = target_book.best_bid
            t_ask = target_book.best_ask
            valid_target_quote = (
                valid_books
                and t_bid is not None
                and t_ask is not None
                and t_bid > 0
                and t_ask > 0
                and t_bid < t_ask
            )
            if valid_target_quote:
                certainty = classify_l2_fill_event(
                    side=order.side,
                    limit_price=order.limit_price,
                    current_bid=t_bid,
                    current_ask=t_ask,
                )
                is_filled = check_price_through_fill(
                    side=order.side,
                    limit_price=order.limit_price,
                    current_bid=t_bid,
                    current_ask=t_ask,
                )
                if is_filled:
                    self.pending_orders.remove(order)
                    order.status = "FILLED"
                    order.filled_quantity = order.order_quantity
                    order.fill_timestamp_ms = current_timestamp_ms
                    order.fill_price = order.limit_price
                    order.fill_certainty = certainty
                    order.target_exit_timestamp_ms = current_timestamp_ms + HOLDING_HORIZON_MS
                    self.open_positions.append(order)
                    self.maker_fills_count += 1

    async def process_buffered_grid_step(
        self,
        grid_timestamp_ms: int,
        up_book: LiveTokenOrderBook,
        down_book: LiveTokenOrderBook,
        market_id: str,
        asset_id: str,
        down_token_id: Optional[str] = None,
    ) -> None:
        """
        Drain events up to grid_timestamp_ms, update books, settle/fill orders,
        and evaluate model inference & regime gating for new passive orders.
        """
        effective_down = down_token_id or getattr(down_book, "asset_id", "")

        # 1. Drain strictly causal point-in-time events
        events_to_apply = self.event_buffer.drain_up_to(grid_timestamp_ms)
        for ev in events_to_apply:
            self.buffered_events_drained_count += 1
            self._apply_parsed_market_data(
                data=ev.data,
                up_book=up_book,
                down_book=down_book,
                now_ms=ev.receipt_timestamp_ms,
                up_token_id=asset_id,
                down_token_id=effective_down,
            )

        # Update book validity flag
        if up_book.is_initialized and down_book.is_initialized and not self.is_book_degraded:
            self.is_book_valid = True
        else:
            self.is_book_valid = False

        # 2. Evaluate pending orders and settling positions
        self._evaluate_pending_orders_and_open_positions(
            current_timestamp_ms=grid_timestamp_ms,
            up_book=up_book,
            down_book=down_book,
            market_id=market_id,
        )

        # 3. Check initialized books and degradation
        if not self.is_book_valid:
            self.uninitialized_book_steps_count += 1
            return

        # 4. Extract top-of-book quotes
        self.eligible_grid_steps_count += 1
        bid = up_book.best_bid
        ask = up_book.best_ask
        bid_size = up_book.bid_size
        ask_size = up_book.ask_size

        down_bid = down_book.best_bid
        down_ask = down_book.best_ask

        has_valid_prices = (
            bid is not None
            and ask is not None
            and bid > 0
            and ask > 0
            and bid < ask
        )
        has_valid_timestamp = (
            up_book.last_update_ms is not None
            and up_book.last_update_ms > 0
            and grid_timestamp_ms >= up_book.last_update_ms
        )

        # If DOWN book quotes are present, validate them as well
        if down_bid is not None or down_ask is not None:
            if down_bid is None or down_ask is None or down_bid <= 0 or down_ask <= 0 or down_bid >= down_ask:
                has_valid_prices = False
            if down_book.last_update_ms is None or down_book.last_update_ms <= 0 or grid_timestamp_ms < down_book.last_update_ms:
                has_valid_timestamp = False

        if not has_valid_prices or not has_valid_timestamp:
            self.unquotable_quotes_count += 1
            reasons = []
            if bid is None or bid <= 0:
                reasons.append("MISSING_OR_NONPOSITIVE_BID")
            if ask is None or ask <= 0:
                reasons.append("MISSING_OR_NONPOSITIVE_ASK")
            if bid is not None and ask is not None and bid >= ask:
                reasons.append("CROSSED_OR_LOCKED_QUOTES")
            if down_bid is not None or down_ask is not None:
                if down_bid is None or down_bid <= 0:
                    reasons.append("MISSING_OR_NONPOSITIVE_DOWN_BID")
                if down_ask is None or down_ask <= 0:
                    reasons.append("MISSING_OR_NONPOSITIVE_DOWN_ASK")
                if down_bid is not None and down_ask is not None and down_bid >= down_ask:
                    reasons.append("CROSSED_OR_LOCKED_DOWN_QUOTES")
            if up_book.last_update_ms is None or up_book.last_update_ms <= 0:
                reasons.append("INVALID_OR_MISSING_TIMESTAMP")
            if up_book.last_update_ms is not None and grid_timestamp_ms < up_book.last_update_ms:
                reasons.append("GRID_EARLIER_THAN_BOOK_UPDATE")
            if down_book.last_update_ms is not None and grid_timestamp_ms < down_book.last_update_ms:
                reasons.append("GRID_EARLIER_THAN_DOWN_BOOK_UPDATE")

            if len(self.unquotable_diagnostics) < 100:
                self.unquotable_diagnostics.append({
                    "session_id": self.session_id,
                    "market_id": market_id,
                    "grid_timestamp_ms": grid_timestamp_ms,
                    "bid": bid,
                    "ask": ask,
                    "reasons": reasons,
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })
            self._check_hard_stops()
            return

        quote_age_ms = float(grid_timestamp_ms - up_book.last_update_ms)
        if down_book.last_update_ms is not None and down_book.last_update_ms > 0:
            down_quote_age_ms = float(grid_timestamp_ms - down_book.last_update_ms)
            quote_age_ms = max(quote_age_ms, down_quote_age_ms)
        self.valid_staleness_observation_count += 1
        self.total_quote_age_ms += quote_age_ms
        if quote_age_ms > MAX_QUOTE_AGE_MS:
            self.stale_quotes_count += 1

        self._check_hard_stops()

        data_gap_ms = float(grid_timestamp_ms - self.last_event_ms) if self.last_event_ms > 0 else 1000.0
        self.last_event_ms = grid_timestamp_ms

        # 5. Add quote to online feature engine
        self.feature_engine.add_grid_sample(
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp_ms=grid_timestamp_ms,
        )

        feat_res = self.feature_engine.compute_current_features_and_sequence()
        if feat_res is None:
            self.warmup_steps_count += 1
            return

        curr_raw_feats, scaled_tensor = feat_res
        scaled_spread = float(scaled_tensor[0, 9, 1])
        scaled_depth_imbal = float(scaled_tensor[0, 9, 10])
        abs_scaled_depth_imbal = abs(scaled_depth_imbal)

        # 6. Model inference
        with torch.no_grad():
            x_torch = torch.from_numpy(scaled_tensor)
            logits = self.model(x_torch)
            probs = torch.softmax(logits, dim=-1).numpy()[0]

        prob_down, prob_flat, prob_up = float(probs[0]), float(probs[1]), float(probs[2])
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        # 7. Predeclared Regime Gates
        rejection_reasons = []
        is_directional = pred_class in (0, 2)
        if self.is_hard_stopped:
            rejection_reasons.append("HARD_STOP_ACTIVE")
        if self.global_abort_ledger.is_globally_aborted():
            rejection_reasons.append("GLOBAL_STRATEGY_ABORT_ACTIVE")
        if not is_directional:
            rejection_reasons.append("FLAT_PREDICTION")
        if confidence < PREDECLARED_CONFIDENCE_THRESHOLD:
            rejection_reasons.append("LOW_CONFIDENCE")
        if scaled_spread > (TRAIN_SPREAD_THRESHOLD + 1e-6):
            rejection_reasons.append("WIDE_SPREAD")
        if abs_scaled_depth_imbal > (TRAIN_DEPTH_IMBALANCE_P90 + 1e-6):
            rejection_reasons.append("EXTREME_DEPTH_IMBALANCE")
        if quote_age_ms > MAX_QUOTE_AGE_MS:
            rejection_reasons.append("STALE_ENTRY_QUOTE")
        if data_gap_ms > MAX_INTER_BURST_GAP_MS and self.observations_logged > 1:
            rejection_reasons.append("INTER_BURST_GAP")

        # Pre-rollover entry blackout window: suppress new orders within 5,000 ms of 5m market expiry
        seconds_into_slot = int(grid_timestamp_ms // 1000) % SLOT_SECONDS
        time_to_rollover_ms = (SLOT_SECONDS - seconds_into_slot) * 1000 - int(grid_timestamp_ms % 1000)
        if 0 < time_to_rollover_ms <= HOLDING_HORIZON_MS:
            rejection_reasons.append("ROLLOVER_BLACKOUT_WINDOW")

        # Specific gate checks for DOWN trades (pred_class == 0)
        if pred_class == 0:
            if down_bid is None or down_ask is None or down_bid <= 0 or down_ask <= 0 or down_bid >= down_ask:
                rejection_reasons.append("INVALID_DOWN_QUOTE")
            if down_book.last_update_ms is None or (grid_timestamp_ms - down_book.last_update_ms) > MAX_QUOTE_AGE_MS:
                if "STALE_ENTRY_QUOTE" not in rejection_reasons:
                    rejection_reasons.append("STALE_ENTRY_QUOTE")

        gate_passed = len(rejection_reasons) == 0
        gate_decision = "EXECUTE" if gate_passed else ("SKIP" if pred_class == 1 else "REJECT")
        self.observations_logged += 1

        if not gate_passed:
            if pred_class == 1:
                self.trades_skipped_flat += 1
            else:
                self.trades_suppressed += 1

            self.shadow_logger.log_observation(
                event_timestamp_ms=grid_timestamp_ms,
                market_id=market_id,
                asset_id=asset_id,
                quote_timestamp_ms=up_book.last_update_ms,
                bid_entry=bid,
                ask_entry=ask,
                depth_imbalance=float(curr_raw_feats[10]),
                scaled_spread=scaled_spread,
                abs_scaled_depth_imbal=abs_scaled_depth_imbal,
                model_probs=(prob_down, prob_flat, prob_up),
                predicted_class=pred_class,
                confidence=confidence,
                burst_id=self.burst_id,
                data_gap_ms=data_gap_ms,
                gate_decision=gate_decision,
                rejection_reasons=rejection_reasons,
                hard_stop_active=self.is_hard_stopped,
            )
        else:
            # Post passive limit order at the touch
            # Class 2 (UP) buys UP token at up_book.best_bid; Class 0 (DOWN) buys DOWN token at down_book.best_bid
            target_token_id = asset_id if pred_class == 2 else effective_down
            target_book = up_book if pred_class == 2 else down_book
            side = "BUY"
            limit_p = float(target_book.best_bid)

            # Deterministic burst clustering: increment burst_id when quiet interval between signals exceeds MAX_INTER_BURST_GAP_MS
            if self.last_signal_timestamp_ms is not None and (grid_timestamp_ms - self.last_signal_timestamp_ms) > MAX_INTER_BURST_GAP_MS:
                self.burst_id += 1
            self.last_signal_timestamp_ms = grid_timestamp_ms

            self.signal_counter += 1
            sig_id = f"sig_{self.session_id}_{grid_timestamp_ms}_{self.signal_counter}"
            self.signals_generated_count += 1

            order = MakerOrder(
                signal_id=sig_id,
                decision_timestamp_ms=grid_timestamp_ms,
                market_id=market_id,
                asset_id=target_token_id,
                side=side,
                predicted_class=pred_class,
                confidence=confidence,
                scaled_spread=scaled_spread,
                abs_scaled_depth_imbal=abs_scaled_depth_imbal,
                burst_id=self.burst_id,
                limit_price=limit_p,
                order_quantity=1.0,
                status="PENDING",
                timeout_timestamp_ms=grid_timestamp_ms + ENTRY_TIMEOUT_MS,
            )
            self.pending_orders.append(order)

            self.shadow_logger.log_observation(
                event_timestamp_ms=grid_timestamp_ms,
                market_id=market_id,
                asset_id=target_token_id,
                quote_timestamp_ms=target_book.last_update_ms,
                bid_entry=float(target_book.best_bid),
                ask_entry=float(target_book.best_ask),
                depth_imbalance=float(curr_raw_feats[10]),
                scaled_spread=scaled_spread,
                abs_scaled_depth_imbal=abs_scaled_depth_imbal,
                model_probs=(prob_down, prob_flat, prob_up),
                predicted_class=pred_class,
                confidence=confidence,
                burst_id=self.burst_id,
                data_gap_ms=data_gap_ms,
                gate_decision="EXECUTE",
                rejection_reasons=[],
                signal_id=sig_id,
                order=order,
                hard_stop_active=self.is_hard_stopped,
            )

    async def run(self) -> Dict[str, Any]:
        """Run prospective shadow session with full robustness and guaranteed finalization."""
        self.is_running = True
        start_time = time.time()
        logger.info("Starting Phase 27 prospective shadow session %s", self.session_id)

        try:
            slot = get_current_5m_slot()
            slug, tokens = await self.token_resolver(slot)
            up_token_id = tokens["UP"]
            down_token_id = tokens["DOWN"]

            up_book = LiveTokenOrderBook(up_token_id)
            down_book = LiveTokenOrderBook(down_token_id)

            reconnect_attempts = 0
            max_reconnects = 5
            base_delay = 1.0

            while self.is_running and reconnect_attempts < max_reconnects:
                reader_task: Optional[asyncio.Task] = None
                try:
                    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                        reconnect_attempts = 0
                        self.is_book_valid = False
                        self.is_book_degraded = False

                        self.event_buffer.clear()
                        while not self.message_queue.empty():
                            try:
                                self.message_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break

                        sub_payload = {
                            "operation": "subscribe",
                            "assets_ids": [up_token_id, down_token_id],
                            "custom_feature_enabled": True,
                        }
                        await ws.send(json.dumps(sub_payload))
                        reader_task = asyncio.create_task(self._ws_reader_task(ws))
                        last_grid_sample_sec = int(time.time())

                        while self.is_running:
                            if reader_task.done():
                                exc = reader_task.exception() if not reader_task.cancelled() else None
                                if exc is not None:
                                    raise exc
                                raise ConnectionClosedOK(None, None)

                            if is_connection_closed(ws):
                                raise ConnectionClosedOK(None, None)

                            now = time.time()
                            elapsed = now - start_time

                            # Stop conditions
                            if self.is_hard_stopped and len(self.open_positions) == 0:
                                logger.info("Hard stop satisfied and positions drained. Stopping cleanly.")
                                self.is_running = False
                                break
                            if self.duration_seconds and elapsed >= self.duration_seconds:
                                logger.info("Session duration reached (%s sec). Stopping cleanly.", self.duration_seconds)
                                self.is_running = False
                                break
                            if self.max_observations and self.observations_logged >= self.max_observations:
                                logger.info("Max observations reached (%s). Stopping cleanly.", self.max_observations)
                                self.is_running = False
                                break

                            # Rollover handling (5m boundary)
                            curr_slot = get_current_5m_slot()
                            if curr_slot != slot:
                                ro_ts_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
                                logger.info("Market rollover: %d -> %d", slot, curr_slot)
                                self.burst_id += 1
                                self.last_signal_timestamp_ms = None
                                try:
                                    next_slug, next_tokens = await self.token_resolver(curr_slot)
                                    # Cancel pending orders from expiring market
                                    for o in list(self.pending_orders):
                                        self.pending_orders.remove(o)
                                        o.status = "CANCELLED_UNFILLED"
                                        self.trade_records.append(TradeRecord(
                                            signal_id=o.signal_id,
                                            record_index=len(self.trade_records),
                                            decision_timestamp_ms=o.decision_timestamp_ms,
                                            market_id=o.market_id,
                                            asset_id=o.asset_id,
                                            burst_id=o.burst_id,
                                            side=o.side,
                                            predicted_class=o.predicted_class,
                                            confidence=o.confidence,
                                            scaled_spread=o.scaled_spread,
                                            abs_scaled_depth_imbal=o.abs_scaled_depth_imbal,
                                            limit_price=o.limit_price,
                                            order_quantity=o.order_quantity,
                                            filled_quantity=0.0,
                                            fill_rate_ratio=0.0,
                                            fill_timestamp_ms=None,
                                            fill_price=None,
                                            exit_timestamp_ms=None,
                                            exit_price=None,
                                            gross_pnl=0.0,
                                            maker_fee=0.0,
                                            taker_fee=0.0,
                                            net_pnl=0.0,
                                            outcome_status="CANCELLED_UNFILLED",
                                            is_filled=False,
                                            is_itt=True,
                                            session_id=self.session_id,
                                        ))

                                    # Settle open in-flight positions as missing exit
                                    for pos in list(self.open_positions):
                                        self.open_positions.remove(pos)
                                        gross_pnl = float(FORCED_LIQUIDATION_SPREAD_PENALTY * pos.filled_quantity)
                                        fee = float(pos.fill_price * pos.filled_quantity * (TAKER_EXIT_FEE_BPS / 10000.0))
                                        net_pnl = float(gross_pnl - fee)
                                        self.trade_records.append(TradeRecord(
                                            signal_id=pos.signal_id,
                                            record_index=len(self.trade_records),
                                            decision_timestamp_ms=pos.decision_timestamp_ms,
                                            market_id=pos.market_id,
                                            asset_id=pos.asset_id,
                                            burst_id=pos.burst_id,
                                            side=pos.side,
                                            predicted_class=pos.predicted_class,
                                            confidence=pos.confidence,
                                            scaled_spread=pos.scaled_spread,
                                            abs_scaled_depth_imbal=pos.abs_scaled_depth_imbal,
                                            limit_price=pos.limit_price,
                                            order_quantity=pos.order_quantity,
                                            filled_quantity=pos.filled_quantity,
                                            fill_rate_ratio=1.0,
                                            fill_timestamp_ms=pos.fill_timestamp_ms,
                                            fill_price=pos.fill_price,
                                            exit_timestamp_ms=int(time.time() * 1000),
                                            exit_price=None,
                                            gross_pnl=gross_pnl,
                                            maker_fee=0.0,
                                            taker_fee=fee,
                                            net_pnl=net_pnl,
                                            outcome_status="INCOMPLETE_MISSING_EXIT",
                                            is_filled=True,
                                            is_itt=True,
                                            session_id=self.session_id,
                                        ))

                                    # Unsubscribe old
                                    try:
                                        await ws.send(json.dumps({"operation": "unsubscribe", "assets_ids": [up_token_id, down_token_id]}))
                                    except Exception:
                                        pass

                                    slot = curr_slot
                                    slug = next_slug
                                    up_token_id = next_tokens["UP"]
                                    down_token_id = next_tokens["DOWN"]
                                    self.rollover_events_count += 1

                                    up_book = LiveTokenOrderBook(up_token_id)
                                    down_book = LiveTokenOrderBook(down_token_id)
                                    self.is_book_valid = False

                                    self.feature_engine.reset()
                                    self.event_buffer.clear()
                                    while not self.message_queue.empty():
                                        try:
                                            self.message_queue.get_nowait()
                                        except asyncio.QueueEmpty:
                                            break

                                    await ws.send(json.dumps({"operation": "subscribe", "assets_ids": [up_token_id, down_token_id], "custom_feature_enabled": True}))
                                    self.rollover_records.append({"timestamp_utc": ro_ts_utc, "new_market": slug, "status": "SUCCESS"})
                                except Exception as ro_err:
                                    logger.warning("Rollover error: %s", ro_err)
                                    self.rollover_records.append({"timestamp_utc": ro_ts_utc, "error": str(ro_err), "status": "FAILED"})

                            self._drain_message_queue_into_buffer(slug, up_book, down_book, max_drain=500)

                            # 1s grid evaluation
                            curr_sec = int(time.time())
                            if curr_sec - last_grid_sample_sec > 5:
                                last_grid_sample_sec = curr_sec - 1

                            while curr_sec > last_grid_sample_sec:
                                last_grid_sample_sec += 1
                                grid_ts_ms = last_grid_sample_sec * 1000

                                self._drain_message_queue_into_buffer(slug, up_book, down_book, max_drain=500)
                                await self.process_buffered_grid_step(
                                    grid_timestamp_ms=grid_ts_ms,
                                    up_book=up_book,
                                    down_book=down_book,
                                    market_id=slug,
                                    asset_id=up_token_id,
                                    down_token_id=down_token_id,
                                )

                            await asyncio.sleep(0.01)

                except (ConnectionClosed, ConnectionError, OSError) as ws_err:
                    reconnect_attempts += 1
                    self.reconnect_count += 1
                    self.is_book_valid = False
                    backoff = min(base_delay * (2 ** (reconnect_attempts - 1)), 15.0) + random.uniform(0, 0.5)
                    logger.warning("WebSocket disconnected (%s). Reconnecting in %.2fs (attempt %d/%d)...", ws_err, backoff, reconnect_attempts, max_reconnects)
                    await asyncio.sleep(backoff)
                finally:
                    if reader_task:
                        if not reader_task.done():
                            reader_task.cancel()
                        try:
                            await reader_task
                        except (asyncio.CancelledError, ConnectionClosed):
                            pass

        except KeyboardInterrupt:
            logger.info("Session interrupted by user.")
        except Exception as fatal_err:
            self.session_fatal_error = str(fatal_err)
            logger.error("Session fatal error: %s", fatal_err, exc_info=True)
        finally:
            manifest = self._finalize_session()
            return manifest

    def _finalize_session(self) -> Dict[str, Any]:
        """
        Guaranteed safe finalization:
        - Flushes pending orders (cancelled) and open positions (missing exit penalty).
        - Closes logger and computes integrity hash.
        - Computes dual-cohort metrics and checks all protocol hurdles.
        - Verifies model weight immutability.
        - Classifies session status: COMPLETE, DEGRADED, ABORTED_HARD_STOP, or FAILED.
        - Records abort in global abort ledger if applicable.
        - Writes manifest and health report.
        """
        if self._is_finalized:
            raise RuntimeError(f"Session {self.session_id} already finalized.")
        self._is_finalized = True

        # 1. Drain pending orders
        final_ts = int(time.time() * 1000)
        for order in list(self.pending_orders):
            self.pending_orders.remove(order)
            order.status = "CANCELLED_UNFILLED"
            self.trade_records.append(TradeRecord(
                signal_id=order.signal_id,
                record_index=len(self.trade_records),
                decision_timestamp_ms=order.decision_timestamp_ms,
                market_id=order.market_id,
                asset_id=order.asset_id,
                burst_id=order.burst_id,
                side=order.side,
                predicted_class=order.predicted_class,
                confidence=order.confidence,
                scaled_spread=order.scaled_spread,
                abs_scaled_depth_imbal=order.abs_scaled_depth_imbal,
                limit_price=order.limit_price,
                order_quantity=order.order_quantity,
                filled_quantity=0.0,
                fill_rate_ratio=0.0,
                fill_timestamp_ms=None,
                fill_price=None,
                exit_timestamp_ms=None,
                exit_price=None,
                gross_pnl=0.0,
                maker_fee=0.0,
                taker_fee=0.0,
                net_pnl=0.0,
                outcome_status="CANCELLED_UNFILLED",
                is_filled=False,
                is_itt=True,
                fill_certainty="UNFILLED",
                session_id=self.session_id,
            ))

        # 2. Drain open positions with missing exit penalty
        for pos in list(self.open_positions):
            self.open_positions.remove(pos)
            gross_pnl = float(FORCED_LIQUIDATION_SPREAD_PENALTY * pos.filled_quantity)
            fee = float(pos.fill_price * pos.filled_quantity * (TAKER_EXIT_FEE_BPS / 10000.0))
            net_pnl = float(gross_pnl - fee)
            self.trade_records.append(TradeRecord(
                signal_id=pos.signal_id,
                record_index=len(self.trade_records),
                decision_timestamp_ms=pos.decision_timestamp_ms,
                market_id=pos.market_id,
                asset_id=pos.asset_id,
                burst_id=pos.burst_id,
                side=pos.side,
                predicted_class=pos.predicted_class,
                confidence=pos.confidence,
                scaled_spread=pos.scaled_spread,
                abs_scaled_depth_imbal=pos.abs_scaled_depth_imbal,
                limit_price=pos.limit_price,
                order_quantity=pos.order_quantity,
                filled_quantity=pos.filled_quantity,
                fill_rate_ratio=1.0,
                fill_timestamp_ms=pos.fill_timestamp_ms,
                fill_price=pos.fill_price,
                exit_timestamp_ms=final_ts,
                exit_price=None,
                gross_pnl=gross_pnl,
                maker_fee=0.0,
                taker_fee=fee,
                net_pnl=net_pnl,
                outcome_status="INCOMPLETE_MISSING_EXIT",
                is_filled=True,
                is_itt=True,
                fill_certainty=getattr(pos, "fill_certainty", None),
                session_id=self.session_id,
            ))

        # 3. Close logger
        logger_manifest = self.shadow_logger.close()

        # 4. Verify model checkpoint immutability
        final_model_hash = sha256_file(self.model_path)
        if self.model_initial_hash != final_model_hash:
            raise RuntimeError("CRITICAL ERROR: Model weights mutated during prospective shadow session!")

        end_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 5. Compute dual-cohort metrics
        dual_metrics = compute_dual_cohort_metrics(
            records=self.trade_records,
            min_filled_trades_target=TARGET_MIN_FILLED_TRADES,
            min_itt_signals_target=TARGET_MIN_ITT_SIGNALS,
            min_bursts_target=TARGET_MIN_BURSTS,
            min_sessions_target=TARGET_MIN_SESSIONS,
            completed_sessions_count=1,
            has_active_hard_stop=self.is_hard_stopped,
        )

        # 6. Status classification
        if self.session_fatal_error is not None:
            session_status = "FAILED"
        elif self.hard_stop_reason is not None:
            session_status = "ABORTED_HARD_STOP"
        elif (
            self.queue_overflow_count > 0
            or self.event_buffer_overflow_count > 0
            or self.malformed_messages_count > 0
            or self.reconnect_count >= 3
        ):
            session_status = "DEGRADED"
        else:
            session_status = "COMPLETE"

        # Record abort in global ledger if hard stop triggered
        if self.hard_stop_reason is not None:
            self.global_abort_ledger.record_abort(
                session_id=self.session_id,
                abort_type=self.hard_stop_type or "STRATEGY_RISK",
                reason=self.hard_stop_reason,
                timestamp_utc=end_utc,
            )

        # 7. Write collector health report
        health_report_path = self.session_dir / "collector_health_report.md"
        self._write_health_report(health_report_path, logger_manifest, end_utc, session_status, dual_metrics)

        # 8. Write serialized trade records audit file
        trade_records_path = self.session_dir / "trade_records.json"
        trade_records_dicts = [asdict(r) for r in self.trade_records]
        with open(trade_records_path, "w", encoding="utf-8") as f:
            json.dump(trade_records_dicts, f, indent=2)
        trade_records_hash = sha256_file(trade_records_path)

        # 9. Write session provenance manifest
        manifest_path = self.session_dir / "session_manifest.json"
        manifest = {
            "session_id": self.session_id,
            "session_type": "GENUINE_PROSPECTIVE_MAKER_SHADOW",
            "session_status": session_status,
            "protocol_version": "2.0_PHASE27_PREDECLARED_FROZEN",
            "hard_stop_triggered": self.hard_stop_reason,
            "hard_stop_type": self.hard_stop_type,
            "start_time_utc": self.start_utc,
            "end_time_utc": end_utc,
            "total_observations_logged": self.observations_logged,
            "warmup_steps_count": self.warmup_steps_count,
            "eligible_grid_steps_count": self.eligible_grid_steps_count,
            "unquotable_quotes_count": self.unquotable_quotes_count,
            "missing_quote_rate": round(self.missing_quote_rate, 6),
            "missing_quote_rate_threshold": MAX_MISSING_QUOTE_RATE,
            "average_quote_age_ms": round(self.average_quote_age_ms, 2),
            "average_quote_staleness_threshold_ms": MAX_AVERAGE_STALENESS_MS,
            "signals_generated_count": self.signals_generated_count,
            "maker_fills_count": self.maker_fills_count,
            "maker_cancellations_count": self.maker_cancellations_count,
            "trades_suppressed": self.trades_suppressed,
            "trades_skipped_flat": self.trades_skipped_flat,
            "ws_messages_received": self.ws_messages_received,
            "json_messages_parsed": self.json_messages_parsed,
            "control_messages_count": self.control_messages_count,
            "protocol_errors_count": self.protocol_errors_count,
            "malformed_messages_count": self.malformed_messages_count,
            "queue_overflow_count": self.queue_overflow_count,
            "max_queue_depth_seen": self.max_queue_depth_seen,
            "reconnect_count": self.reconnect_count,
            "rollover_events_count": self.rollover_events_count,
            "fatal_error": self.session_fatal_error,
            "model_checkpoint_sha256": final_model_hash,
            "log_file": logger_manifest["log_file"],
            "log_sha256": logger_manifest["log_sha256"],
            "rolling_chain_hash": logger_manifest["rolling_chain_hash"],
            "health_report_sha256": sha256_file(health_report_path),
            "trade_records_file": str(trade_records_path.name),
            "trade_records_sha256": trade_records_hash,
            "trade_records_count": len(self.trade_records),
            "locked_test_set_touched": False,
            "real_capital_used": False,
            "exchange_orders_sent": False,
            "dual_cohort_metrics": dual_metrics,
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Session %s finalized [%s]. Saved to %s", self.session_id, session_status, self.session_dir)
        return manifest

    def _write_health_report(
        self,
        path: Path,
        logger_manifest: Dict[str, Any],
        end_utc: str,
        session_status: str,
        dual_metrics: Dict[str, Any],
    ) -> None:
        """Generate human-readable markdown health report and dual-cohort scoreboard."""
        status_badge = (
            "**HEALTHY (COMPLETE)**" if session_status == "COMPLETE"
            else ("**ABORTED (HARD STOP TRIGGERED)**" if session_status == "ABORTED_HARD_STOP"
            else ("**DEGRADED**" if session_status == "DEGRADED" else "**FAILED**"))
        )

        itt = dual_metrics.get("itt_cohort", {})
        filled = dual_metrics.get("filled_cohort", {})
        dir_b = dual_metrics.get("directional_balance", {})

        md = [
            f"# Phase 27 Prospective Shadow Collector Health Report — Session `{self.session_id}`\n",
            "> [!IMPORTANT]",
            "> **RESEARCH-ONLY PROSPECTIVE TELEMETRY**: Zero real capital deployed; zero live orders sent to exchange.",
            "> All hypothetical returns represent simulated passive limit orders at touch with strict price-through fills and 5 bps taker exit fees.\n",
            "---\n",
            f"## 1. Session Status: {status_badge}\n",
            "| Telemetry Metric | Session Value | Quality Benchmark |",
            "| :--- | :---: | :--- |",
            "| **Protocol Version** | `2.0_PHASE27_PREDECLARED_FROZEN` | Locked Ratified Protocol |",
            f"| **Start Timestamp (UTC)** | `{self.start_utc}` | Recorded |",
            f"| **End Timestamp (UTC)** | `{end_utc}` | Recorded |",
            f"| **Session Status** | `{session_status}` | Operational certification |",
            f"| **Raw WebSocket Frames** | `{self.ws_messages_received:,}` | Active feed connectivity |",
            f"| **JSON Payloads Parsed** | `{self.json_messages_parsed:,}` | Normal market data |",
            f"| **Queue Overflows** | `{self.queue_overflow_count:,}` | 0 required for strict continuity |",
            f"| **Peak Queue Depth** | `{self.max_queue_depth_seen:,}` | Capacity: {self.queue_maxsize:,} |",
            f"| **Reconnect Attempts** | `{self.reconnect_count:,}` | Resubscription & snapshot rebuild |",
            f"| **Market Rollovers** | `{self.rollover_events_count:,}` | Feature engine cleanly reset |",
            f"| **Eligible Grid Steps** | `{self.eligible_grid_steps_count:,}` | Decision denominator |",
            f"| **Missing Quote Rate** | `{self.missing_quote_rate * 100:.2f}%` | Threshold: <= 25.0% |",
            f"| **Average Quote Staleness** | `{self.average_quote_age_ms:.2f} ms` | Threshold: <= 2000.0 ms |",
            f"| **Signals Generated (ITT)** | `{self.signals_generated_count:,}` | Target: >= 600 across sessions |",
            f"| **Maker Fills** | `{self.maker_fills_count:,}` | Strict price-through fills |",
            f"| **Maker Cancellations** | `{self.maker_cancellations_count:,}` | Expired at 2,000 ms timeout |",
            f"| **Trades Suppressed** | `{self.trades_suppressed:,}` | Strict regime gating |",
            f"| **Trades Skipped (Flat)** | `{self.trades_skipped_flat:,}` | Class 1 predictions |\n",
            "---\n",
            "## 2. Dual-Cohort Performance Scoreboard\n",
            "| Metric | ITT Master Cohort | Filled-Trade Cohort | Predeclared Hurdle | Hurdle Status |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| **Denominator** | `{itt.get('denominator_signals', 0)}` signals | `{filled.get('denominator_trades', 0)}` trades | >= 600 ITT / >= 400 Filled | {'PASS' if itt.get('denominator_signals', 0) >= 600 and filled.get('denominator_trades', 0) >= 400 else 'SAMPLE_INCOMPLETE'} |",
            f"| **Mean Net PnL** | `{itt.get('mean_net_pnl_per_signal', 0.0):.6f}` | `{filled.get('mean_net_pnl_per_trade', 0.0):.6f}` | >= +0.0025 ITT / >= +0.0030 Filled | {'PASS' if itt.get('hurdle_passed') and filled.get('hurdle_passed') else 'FAIL'} |",
            f"| **Profit Factor** | N/A | `{filled.get('profit_factor', 0.0):.4f}` | >= 1.25 (target 1.30) | {'PASS' if filled.get('profit_factor_floor_passed') else 'FAIL'} |",
            f"| **Decisive Win Rate** | N/A | `{filled.get('win_rate_decisive', 0.0) * 100:.2f}%` | >= 50.0% | {'PASS' if filled.get('decisive_win_rate_passed') else 'FAIL'} |",
            f"| **Overall Win Rate** | N/A | `{filled.get('win_rate_overall', 0.0) * 100:.2f}%` | >= 48.0% | {'PASS' if filled.get('overall_win_rate_passed') else 'FAIL'} |",
            f"| **Cluster-Robust t (t_CR)** | `{itt.get('cluster_robust_t', 0.0):.2f}` | `{filled.get('cluster_robust_t', 0.0):.2f}` | >= 2.0 | {'PASS' if filled.get('t_stat_passed') else 'FAIL'} |",
            f"| **Empirical N_eff** | `{itt.get('empirical_n_eff', 0.0):.1f}` | `{filled.get('empirical_n_eff', 0.0):.1f}` | Target >= 40.0 | {'PASS' if filled.get('empirical_n_eff', 0.0) >= 40.0 else 'EVALUATED'} |",
            f"| **Directional Asymmetry** | `{dir_b.get('directional_asymmetry_index', 0.0):.4f}` | `{dir_b.get('directional_asymmetry_index', 0.0):.4f}` | <= 0.35 | {'PASS' if dir_b.get('balance_passed') else 'FAIL'} |\n",
            "### L2 Fill Certainty Scenarios (Conservative vs Optimistic)\n",
            "| Scenario | Filled Trades | ITT Mean Net PnL | ITT Cluster-Robust t | Description |",
            "| :--- | :---: | :---: | :---: | :--- |",
            f"| **Conservative** | `{dual_metrics.get('fill_certainty_scenarios', {}).get('conservative', {}).get('filled_count', 0)}` | `{dual_metrics.get('fill_certainty_scenarios', {}).get('conservative', {}).get('itt_mean_net_pnl', 0.0):.6f}` | `{dual_metrics.get('fill_certainty_scenarios', {}).get('conservative', {}).get('itt_cluster_robust_t', 0.0):.2f}` | Confirmed Price-Through only (`Ask <= Limit`) |",
            f"| **Optimistic** | `{dual_metrics.get('fill_certainty_scenarios', {}).get('optimistic', {}).get('filled_count', 0)}` | `{dual_metrics.get('fill_certainty_scenarios', {}).get('optimistic', {}).get('itt_mean_net_pnl', 0.0):.6f}` | `{dual_metrics.get('fill_certainty_scenarios', {}).get('optimistic', {}).get('itt_cluster_robust_t', 0.0):.2f}` | Confirmed Price-Through + Quote Retreats |",
            f"\n*L2 Breakdown*: Confirmed Price-Through: {dual_metrics.get('fill_certainty_scenarios', {}).get('breakdown', {}).get('confirmed_price_through_count', 0)} | Ambiguous Quote Retreats: {dual_metrics.get('fill_certainty_scenarios', {}).get('breakdown', {}).get('ambiguous_quote_retreat_count', 0)} | Touch Rejected: {dual_metrics.get('fill_certainty_scenarios', {}).get('breakdown', {}).get('touch_rejected_count', 0)} | Unfilled: {dual_metrics.get('fill_certainty_scenarios', {}).get('breakdown', {}).get('unfilled_count', 0)}.\n",
            "---\n",
            "## 3. Cryptographic Invariants & Provenance\n",
            "- [x] **Zero Live Orders**: Read-only WebSocket telemetry; zero capital deployed.",
            "- [x] **Model Weights Frozen**: SmallLSTM weights verified against SHA-256 reference.",
            "- [x] **Locked Test Isolation**: `test_scaled.npz` was never opened or read.",
            f"- [x] **Observation Log File**: `{logger_manifest['log_file']}` (SHA-256: `{logger_manifest['log_sha256'][:16]}...`).",
            f"- [x] **Rolling Chain Digest**: `{logger_manifest['rolling_chain_hash'][:16]}...`.\n",
        ]

        if self.hard_stop_reason:
            md.extend([
                "---\n",
                "## 4. Hard Stop Circuit Breaker Activation\n",
                f"- **Trigger Code**: `{self.hard_stop_reason}`\n",
                f"- **Abort Classification**: `{self.hard_stop_type}`\n",
                "- **Enforcement**: New orders halted immediately; pending orders cancelled; in-flight positions settled.\n",
            ])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")


# -----------------------------------------------------------------------------
# Section 4: CLI Entrypoint
# -----------------------------------------------------------------------------

def setup_utc_logging(level: int = logging.INFO) -> None:
    """Configure root logging with explicit UTC timestamps."""
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=level,
        format="%(asctime)s UTC [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Phase 27 Prospective Maker Shadow Session")
    parser.add_argument("--duration-seconds", type=int, default=None, help="Stop session after N seconds")
    parser.add_argument("--max-observations", type=int, default=None, help="Stop session after N grid observations")
    parser.add_argument("--session-id", type=str, default=None, help="Custom session ID string")
    parser.add_argument("--out-dir", type=Path, default=PROSPECTIVE_BASE_DIR, help="Base output directory")
    args = parser.parse_args()

    setup_utc_logging(level=logging.INFO)

    session = Phase27ProspectiveSession(
        output_base_dir=args.out_dir,
        session_id=args.session_id,
        duration_seconds=args.duration_seconds,
        max_observations=args.max_observations,
    )
    manifest = asyncio.run(session.run())
    print(f"Session {session.session_id} finalized with status: {manifest['session_status']}")


if __name__ == "__main__":
    main()
