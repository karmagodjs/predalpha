"""
Phase 26 Prospective Shadow Collector — Live Forward Telemetry & Execution Risk.

Connects to the Polymarket CLOB WebSocket and Gamma REST API to consume a genuine
real-time stream of verified bid/ask quotes and order book updates for 5-minute Bitcoin markets.

Hotfix & Robustness Invariants:
1. Message Classification: Safely parses empty frames, heartbeat/control strings, binary,
   and JSON payloads. On JSONDecodeError, records bounded sanitized diagnostics without crashing.
2. Slow-Consumer Decoupling: Separates fast socket receiving from feature computation and inference
   using an asyncio.Queue with explicit backpressure, queue-depth telemetry, and degraded book flags.
3. Resilient Reconnects: Implements bounded exponential backoff with jitter and requires fresh order book
   snapshots before re-enabling trading decisions.
4. Clean Market Rollover: Resolves new 5m market token IDs, updates subscriptions, resets order books,
   and resets the causal feature engine (preventing cross-market quote contamination).
5. Guaranteed Finalization: try/finally ensures that sessions (even on error or interrupt) flush pending
   positions as INCOMPLETE_MISSING_EXIT, close sockets, and generate manifests with accurate status
   (COMPLETE, DEGRADED, or FAILED).
6. Non-Negotiable Safety: Zero exchange orders, zero capital, frozen model weights, frozen Phase 25 gates,
   and zero access to locked test dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request
import uuid

import numpy as np
import torch
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError
try:
    from websockets.protocol import State as WebSocketState
except ImportError:
    try:
        from websockets.connection import State as WebSocketState
    except ImportError:
        WebSocketState = None


def is_connection_closed(ws: Any) -> bool:
    """
    Check if a WebSocket connection is closed or closing across websockets library versions.
    Compatible with websockets v11+ (ws.state in CLOSING/CLOSED) and legacy/mock (ws.closed).
    """
    if ws is None:
        return True
    if hasattr(ws, "state"):
        state = ws.state
        if WebSocketState is not None:
            return state in (WebSocketState.CLOSING, WebSocketState.CLOSED)
        state_str = str(getattr(state, "name", state)).upper()
        return "CLOSED" in state_str or "CLOSING" in state_str
    if hasattr(ws, "closed"):
        return bool(ws.closed)
    return False

from pipeline_v2.models.phase26_forward_shadow import (
    ForwardShadowLogger,
    MODEL_PATH,
    PREDECLARED_CONFIDENCE_THRESHOLD,
    TRAIN_SPREAD_THRESHOLD,
    TRAIN_DEPTH_IMBALANCE_P90,
    LOCKED_TEST_PATH,
    sha256_file,
)
from pipeline_v2.models.recurrent_models import SmallLSTMModule

logger = logging.getLogger("pipeline_v2.models.phase26_prospective")

ROOT = Path(__file__).resolve().parents[2]
SCALER_PARAMS_PATH = ROOT / "data" / "clean_v2" / "07_scaled" / "expanded_collection" / "scaler_params.json"
PROSPECTIVE_BASE_DIR = ROOT / "data" / "models" / "phase26" / "prospective"

GAMMA_URL = "https://gamma-api.polymarket.com"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
SLOT_SECONDS = 300
QUEUE_MAXSIZE = 10000


# -----------------------------------------------------------------------------
# Section 1: Message Classification & Safe Parsing
# -----------------------------------------------------------------------------

def classify_and_parse_ws_message(
    msg: Union[str, bytes],
) -> Tuple[str, Optional[Any], Optional[str]]:
    """
    Safely classify and parse incoming WebSocket frames before processing.

    Returns:
        (category, parsed_data, error_detail)
        where category is:
          - 'JSON': parsed_data is dict or list
          - 'EMPTY': empty string or whitespace frame
          - 'CONTROL': heartbeat / control string (e.g. 'PONG', 'PING')
          - 'PROTOCOL_ERROR': exchange protocol status or rejection string (e.g. 'INVALID OPERATION')
          - 'BINARY': non-utf8 binary frame
          - 'MALFORMED': non-empty text that failed json.loads
    """
    if isinstance(msg, bytes):
        try:
            msg = msg.decode("utf-8")
        except UnicodeDecodeError as err:
            return "BINARY", None, f"UnicodeDecodeError: {err}"

    if not isinstance(msg, str):
        return "MALFORMED", None, f"Unexpected message type: {type(msg).__name__}"

    cleaned = msg.strip()
    if not cleaned:
        return "EMPTY", None, None

    # Check for known control / heartbeat text frames
    upper = cleaned.upper()
    if upper in ("PONG", "PING", "OK", "HEARTBEAT", "PONG\n", "PING\n"):
        return "CONTROL", cleaned, None

    # Check for known exchange protocol rejection / status strings
    if upper in ("INVALID OPERATION", "NO NEW ASSETS", "ALREADY SUBSCRIBED"):
        return "PROTOCOL_ERROR", cleaned, f"Polymarket protocol message: {cleaned}"

    try:
        data = json.loads(cleaned)
        return "JSON", data, None
    except json.JSONDecodeError as exc:
        return "MALFORMED", None, f"JSONDecodeError: {exc}"


# -----------------------------------------------------------------------------
# Section 2: Live Market Discovery & Order Book State Tracking
# -----------------------------------------------------------------------------

def get_current_5m_slot(timestamp_sec: Optional[float] = None) -> int:
    """Return the floor 5-minute epoch timestamp."""
    ts = timestamp_sec if timestamp_sec is not None else time.time()
    return (int(ts) // SLOT_SECONDS) * SLOT_SECONDS


def fetch_market_tokens_for_slot(slot: int, timeout: float = 10.0) -> Tuple[str, Dict[str, str]]:
    """
    Query Gamma REST API for the 5-minute Bitcoin Up/Down market at given slot.
    Returns (slug, tokens_dict {'UP': token_id, 'DOWN': token_id}).
    """
    slug = f"btc-updown-5m-{slot}"
    url = f"{GAMMA_URL}/events/slug/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": "PredAlpha/1.0"})

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        event = json.loads(resp.read().decode("utf-8"))

    markets = event.get("markets", [])
    if not markets:
        raise ValueError(f"No markets found for slug {slug}")

    for m in markets:
        outcomes = m.get("outcomes", [])
        token_ids = m.get("clobTokenIds", [])

        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)

        if len(outcomes) != len(token_ids):
            continue

        omap = {str(o).strip().lower(): str(t) for o, t in zip(outcomes, token_ids)}
        if "up" in omap and "down" in omap:
            return slug, {"UP": omap["up"], "DOWN": omap["down"]}

    raise ValueError(f"Could not resolve UP/DOWN token mapping for {slug}")


class LiveTokenOrderBook:
    """
    Tracks order book levels (bids and asks) for a single token.
    Applies initial snapshots and incremental price_change delta updates.
    Synchronizes with exchange-reported canonical best_bid and best_ask to eliminate
    crossed books or stale top-of-book levels during high volatility.
    """

    def __init__(self, asset_id: str):
        self.asset_id = str(asset_id)
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.exchange_best_bid: Optional[float] = None
        self.exchange_best_ask: Optional[float] = None
        self.last_update_ms: int = 0
        self.is_initialized: bool = False

    def reset(self) -> None:
        """Clear book state upon market rollover or reconnect."""
        self.bids.clear()
        self.asks.clear()
        self.exchange_best_bid = None
        self.exchange_best_ask = None
        self.last_update_ms = 0
        self.is_initialized = False

    def apply_book_event(self, data: Dict[str, Any], timestamp_ms: int) -> None:
        """Initialize or reset book from full snapshot."""
        self.bids.clear()
        self.asks.clear()
        for lvl in data.get("bids", []):
            p, s = float(lvl["price"]), float(lvl["size"])
            if s > 0:
                self.bids[p] = s
        for lvl in data.get("asks", []):
            p, s = float(lvl["price"]), float(lvl["size"])
            if s > 0:
                self.asks[p] = s
        self.exchange_best_bid = max(self.bids.keys()) if self.bids else None
        self.exchange_best_ask = min(self.asks.keys()) if self.asks else None
        self.last_update_ms = timestamp_ms
        self.is_initialized = True

    def apply_price_change(self, change: Dict[str, Any], timestamp_ms: int) -> None:
        """Apply incremental delta update and synchronize top-of-book."""
        price = float(change.get("price", 0.0))
        size = float(change.get("size", 0.0))
        side = change.get("side")

        if side == "BUY":
            if size <= 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size
        elif side == "SELL":
            if size <= 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size

        # Update canonical best_bid and best_ask if reported by exchange
        bb_raw = change.get("best_bid")
        if bb_raw is not None:
            try:
                bb_val = float(bb_raw)
                if bb_val > 0:
                    self.exchange_best_bid = bb_val
                    # Prune stale bids strictly higher than exchange best_bid
                    for p in list(self.bids.keys()):
                        if p > bb_val + 1e-9:
                            self.bids.pop(p, None)
            except (ValueError, TypeError):
                pass
        else:
            self.exchange_best_bid = max(self.bids.keys()) if self.bids else None

        ba_raw = change.get("best_ask")
        if ba_raw is not None:
            try:
                ba_val = float(ba_raw)
                if ba_val > 0:
                    self.exchange_best_ask = ba_val
                    # Prune stale asks strictly lower than exchange best_ask
                    for p in list(self.asks.keys()):
                        if p < ba_val - 1e-9:
                            self.asks.pop(p, None)
            except (ValueError, TypeError):
                pass
        else:
            self.exchange_best_ask = min(self.asks.keys()) if self.asks else None

        # Prune crossed levels if both best_bid and best_ask are present
        eff_bb = self.exchange_best_bid or (max(self.bids.keys()) if self.bids else None)
        eff_ba = self.exchange_best_ask or (min(self.asks.keys()) if self.asks else None)
        if eff_bb is not None and eff_ba is not None and eff_bb < eff_ba:
            for p in list(self.bids.keys()):
                if p >= eff_ba - 1e-9:
                    self.bids.pop(p, None)
            for p in list(self.asks.keys()):
                if p <= eff_bb + 1e-9:
                    self.asks.pop(p, None)

        self.last_update_ms = timestamp_ms

    @property
    def best_bid(self) -> Optional[float]:
        if self.bids:
            max_bid = max(self.bids.keys())
            if self.exchange_best_bid is not None:
                return max(max_bid, self.exchange_best_bid)
            return max_bid
        return self.exchange_best_bid

    @property
    def best_ask(self) -> Optional[float]:
        if self.asks:
            min_ask = min(self.asks.keys())
            if self.exchange_best_ask is not None:
                return min(min_ask, self.exchange_best_ask)
            return min_ask
        return self.exchange_best_ask

    @property
    def bid_size(self) -> float:
        bb = self.best_bid
        if bb is None:
            return 0.0
        return self.bids.get(bb, 100.0)

    @property
    def ask_size(self) -> float:
        ba = self.best_ask
        if ba is None:
            return 0.0
        return self.asks.get(ba, 100.0)

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        return (bb + ba) / 2.0 if (bb is not None and ba is not None) else None

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        return (ba - bb) if (bb is not None and ba is not None) else None


# -----------------------------------------------------------------------------
# Section 3: Online Causal Feature Engineering & Scaling
# -----------------------------------------------------------------------------

class OnlineCausalFeatureEngine:
    """
    Computes the exact 11 causal features and scales them for SmallLSTM inference.
    Maintains a rolling 1-second grid buffer (minimum 10 steps) with zero lookahead.
    """

    FEATURE_NAMES = [
        "mid_price",
        "spread",
        "spread_bps",
        "mid_return_1s",
        "mid_return_3s",
        "mid_return_5s",
        "mid_volatility_5s",
        "bid_change_1s",
        "ask_change_1s",
        "microprice",
        "depth_imbalance",
    ]

    def __init__(self, scaler_params_path: Path = SCALER_PARAMS_PATH):
        if not scaler_params_path.exists():
            raise FileNotFoundError(f"Scaler parameters not found at {scaler_params_path}")
        with open(scaler_params_path, "r", encoding="utf-8") as f:
            sp = json.load(f)

        self.centers = np.array([sp["centers"][name] for name in self.FEATURE_NAMES], dtype=np.float32)
        self.scales = np.array([sp["scales"][name] for name in self.FEATURE_NAMES], dtype=np.float32)
        self.scales = np.where(self.scales < 1e-8, 1.0, self.scales)

        # Buffer of 1s grid points: list of dicts with 'bid', 'ask', 'bid_size', 'ask_size', 'timestamp_ms'
        self.grid_history: List[Dict[str, float]] = []

    def reset(self) -> None:
        """Clear history on market rollover to prevent cross-market contamination."""
        self.grid_history.clear()

    def add_grid_sample(self, bid: float, ask: float, bid_size: float, ask_size: float, timestamp_ms: int) -> None:
        """Add a 1-second sampled quote."""
        self.grid_history.append({
            "bid": float(bid),
            "ask": float(ask),
            "bid_size": float(bid_size),
            "ask_size": float(ask_size),
            "timestamp_ms": int(timestamp_ms),
        })
        # Keep at most 30 steps of history for rolling return/volatility computations
        if len(self.grid_history) > 30:
            self.grid_history.pop(0)

    def is_warmed_up(self) -> bool:
        """Requires at least 10 consecutive steps for sequence length L=10 and 5s rolling volatility."""
        return len(self.grid_history) >= 10

    def compute_current_features_and_sequence(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute unscaled features for current step, and scaled sequence tensor (1, 10, 11).
        Returns (current_raw_features, scaled_sequence_tensor) or None if not warmed up.
        """
        if not self.is_warmed_up():
            return None

        # Build feature vectors for the last 10 steps
        n_steps = len(self.grid_history)
        features_seq = np.zeros((10, 11), dtype=np.float32)

        for k in range(10):
            idx = n_steps - 10 + k
            curr = self.grid_history[idx]
            bid = curr["bid"]
            ask = curr["ask"]
            bid_size = curr["bid_size"]
            ask_size = curr["ask_size"]

            mid = (bid + ask) / 2.0
            spread = ask - bid
            spread_bps = (spread / max(mid, 1e-6)) * 10000.0

            # 1s return
            prev1 = self.grid_history[idx - 1] if idx >= 1 else curr
            mid_prev1 = (prev1["bid"] + prev1["ask"]) / 2.0
            mid_ret_1s = (mid - mid_prev1) / max(mid_prev1, 1e-6)

            # 3s return
            prev3 = self.grid_history[idx - 3] if idx >= 3 else curr
            mid_prev3 = (prev3["bid"] + prev3["ask"]) / 2.0
            mid_ret_3s = (mid - mid_prev3) / max(mid_prev3, 1e-6)

            # 5s return
            prev5 = self.grid_history[idx - 5] if idx >= 5 else curr
            mid_prev5 = (prev5["bid"] + prev5["ask"]) / 2.0
            mid_ret_5s = (mid - mid_prev5) / max(mid_prev5, 1e-6)

            # 5s volatility (std of 1s returns in [idx-4, ..., idx])
            if idx >= 4:
                window_rets = []
                for w in range(idx - 4, idx + 1):
                    p_curr = (self.grid_history[w]["bid"] + self.grid_history[w]["ask"]) / 2.0
                    p_prior = (self.grid_history[w - 1]["bid"] + self.grid_history[w - 1]["ask"]) / 2.0 if w >= 1 else p_curr
                    window_rets.append((p_curr - p_prior) / max(p_prior, 1e-6))
                mid_vol_5s = float(np.std(window_rets, ddof=1)) if len(window_rets) > 1 else 0.0
            else:
                mid_vol_5s = 0.0

            bid_change_1s = bid - prev1["bid"]
            ask_change_1s = ask - prev1["ask"]

            tot_depth = max(bid_size + ask_size, 1e-6)
            microprice = (bid * ask_size + ask * bid_size) / tot_depth
            depth_imbalance = (bid_size - ask_size) / tot_depth

            feat_row = np.array([
                mid, spread, spread_bps, mid_ret_1s, mid_ret_3s, mid_ret_5s,
                mid_vol_5s, bid_change_1s, ask_change_1s, microprice, depth_imbalance
            ], dtype=np.float32)

            features_seq[k] = feat_row

        current_raw_features = features_seq[-1].copy()

        # Standard scale the length-10 sequence: (features - center) / scale
        scaled_seq = (features_seq - self.centers) / self.scales
        scaled_tensor = np.expand_dims(scaled_seq, axis=0)  # (1, 10, 11)

        return current_raw_features, scaled_tensor


# -----------------------------------------------------------------------------
# Section 4: Prospective Shadow Session Runner
# -----------------------------------------------------------------------------

class ProspectiveShadowSession:
    """
    Manages a live prospective shadow recording session.
    Features:
    - Decoupled fast reader task preventing slow consumer disconnects.
    - Robust frame parsing handling empty/control/malformed frames gracefully.
    - Safe market rollover resetting order books and feature engines.
    - Exponential backoff reconnects requiring fresh book snapshots before decisions.
    - Guaranteed try/finally finalization writing manifests and health reports on all paths.
    """

    def __init__(
        self,
        output_base_dir: Path = PROSPECTIVE_BASE_DIR,
        model_path: Path = MODEL_PATH,
        session_id: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        max_observations: Optional[int] = None,
        queue_maxsize: int = QUEUE_MAXSIZE,
    ):
        # Strict locked test set isolation check
        if not LOCKED_TEST_PATH.exists():
            raise FileNotFoundError(f"Safety check failed: locked test path {LOCKED_TEST_PATH} not found.")

        self.session_id = session_id or f"prospective_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.session_dir = output_base_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.duration_seconds = duration_seconds
        self.max_observations = max_observations
        self.queue_maxsize = queue_maxsize

        # Load frozen SmallLSTM model
        self.model_path = model_path
        self.model_initial_hash = sha256_file(model_path)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = SmallLSTMModule(input_size=11, hidden_size=32, num_layers=1, num_classes=3)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.feature_engine = OnlineCausalFeatureEngine()
        self.shadow_logger = ForwardShadowLogger(output_dir=self.session_dir, run_id=self.session_id)

        # Pending hypothetical positions: waiting for 5s horizon
        self.pending_positions: List[Dict[str, Any]] = []

        # Decoupled message queue
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)

        # Operational state flags
        self.is_running = False
        self.is_book_valid = False
        self.is_book_degraded = False

        # Session telemetry
        self.start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.ws_messages_received = 0
        self.json_messages_parsed = 0
        self.control_messages_count = 0
        self.protocol_errors_count = 0
        self.malformed_messages_count = 0
        self.queue_overflow_count = 0
        self.max_queue_depth_seen = 0
        self.reconnect_count = 0
        self.rollover_events_count = 0

        self.observations_logged = 0
        self.trades_executed = 0
        self.trades_suppressed = 0
        self.trades_skipped_flat = 0
        self.stale_quotes_count = 0
        self.warmup_steps_count = 0
        self.unquotable_quotes_count = 0
        self.uninitialized_book_steps_count = 0
        self.burst_id = 0
        self.last_event_ms = 0
        self.session_fatal_error: Optional[str] = None
        self.malformed_samples: List[Dict[str, Any]] = []
        self.rollover_records: List[Dict[str, Any]] = []

        # Predeclared hard stopping triggers (from prospective_protocol.json)
        self.consecutive_losses = 0
        self.consecutive_losses_limit = 8
        self.cum_net_pnl = 0.0
        self.peak_net_pnl = 0.0
        self.max_drawdown_seen = 0.0
        self.max_drawdown_stop_loss_units = 2.5
        self.is_hard_stopped = False
        self.hard_stop_reason: Optional[str] = None

    async def _ws_reader_task(self, ws: Any) -> None:
        """
        Fast dedicated WebSocket receiver.
        Only reads frames and immediately pushes them to the queue without blocking.
        Drains network buffers at line speed, eliminating slow-consumer close code 1013.
        """
        while self.is_running:
            try:
                msg = await ws.recv()
                self.ws_messages_received += 1

                # Track queue depth telemetry
                depth = self.message_queue.qsize()
                if depth > self.max_queue_depth_seen:
                    self.max_queue_depth_seen = depth

                try:
                    self.message_queue.put_nowait(msg)
                except asyncio.QueueFull:
                    self.queue_overflow_count += 1
                    self.is_book_degraded = True
                    logger.warning(
                        "WebSocket queue overflow (depth=%d). Marking order book degraded until fresh snapshot.",
                        depth
                    )
            except asyncio.CancelledError:
                raise
            except ConnectionClosed as conn_err:
                logger.info("WebSocket connection closed in reader task: %s", conn_err)
                raise
            except (ConnectionError, OSError) as net_err:
                logger.warning("WebSocket network error in reader task: %s", net_err)
                raise
            except Exception as recv_err:
                logger.warning("ws_reader_task unexpected error: %s", recv_err)
                raise

    async def run(self) -> Dict[str, Any]:
        """
        Run the prospective shadow session with full robustness and guaranteed finalization.
        """
        self.is_running = True
        start_time = time.time()
        logger.info("Starting prospective shadow session %s", self.session_id)

        try:
            # 1. Discover active market
            slot = get_current_5m_slot()
            slug, tokens = await asyncio.to_thread(fetch_market_tokens_for_slot, slot)
            logger.info("Discovered active market: %s | UP: %s | DOWN: %s", slug, tokens["UP"][:10], tokens["DOWN"][:10])

            up_token_id = tokens["UP"]
            down_token_id = tokens["DOWN"]

            up_book = LiveTokenOrderBook(up_token_id)
            down_book = LiveTokenOrderBook(down_token_id)

            reconnect_attempts = 0
            max_reconnects = 5
            base_delay = 1.0
            max_delay = 15.0
            jitter = 0.5

            while self.is_running and reconnect_attempts < max_reconnects:
                reader_task: Optional[asyncio.Task] = None
                try:
                    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                        reconnect_attempts = 0
                        self.is_book_valid = False
                        self.is_book_degraded = False

                        # Clear queue of any stale messages from prior disconnect
                        while not self.message_queue.empty():
                            try:
                                self.message_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break

                        # Send initial subscription
                        sub_payload = {
                            "operation": "subscribe",
                            "assets_ids": [up_token_id, down_token_id],
                            "custom_feature_enabled": True,
                        }
                        await ws.send(json.dumps(sub_payload))
                        logger.info("Subscribed to WebSocket feed for market %s", slug)

                        # Start decoupled receiver task
                        reader_task = asyncio.create_task(self._ws_reader_task(ws))
                        last_grid_sample_sec = int(time.time())
                        rollover_pending_snapshot_since: Optional[float] = None

                        while self.is_running:
                            # 1. Check if background reader task ended (connection dropped or error)
                            if reader_task.done():
                                exc = reader_task.exception() if not reader_task.cancelled() else None
                                if exc is not None:
                                    logger.warning("WebSocket reader task terminated with error: %s", exc)
                                    raise exc
                                else:
                                    logger.info("WebSocket reader task completed.")
                                    raise ConnectionClosedOK(None, None)

                            # 2. Check if connection state is closed or closing
                            if is_connection_closed(ws):
                                logger.info("WebSocket connection state is closed/closing; triggering reconnect.")
                                raise ConnectionClosedOK(None, None)

                            now = time.time()
                            elapsed = now - start_time

                            # Check session termination criteria
                            if self.is_hard_stopped and len(self.pending_positions) == 0:
                                logger.info(
                                    "Hard stopping abort trigger satisfied (%s) and pending positions drained. Stopping cleanly.",
                                    self.hard_stop_reason
                                )
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

                            # Check market slot rollover (5m boundary)
                            curr_slot = get_current_5m_slot()
                            if curr_slot != slot:
                                rollover_ts_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
                                logger.info("Market rollover detected: slot %d -> %d. Resolving new market...", slot, curr_slot)
                                try:
                                    next_slug, next_tokens = await asyncio.to_thread(fetch_market_tokens_for_slot, curr_slot)
                                    if "UP" not in next_tokens or "DOWN" not in next_tokens:
                                        raise ValueError(f"Incomplete token mapping for {next_slug}: {next_tokens}")

                                    # Close any pending hypothetical positions from the expiring market
                                    for pos in list(self.pending_positions):
                                        pos["has_exit_quote"] = False
                                        pos["exit_timestamp_ms"] = None
                                        pos["bid_exit"] = None
                                        pos["ask_exit"] = None
                                        self._log_and_complete_position(pos)
                                    self.pending_positions.clear()

                                    # Unsubscribe previous tokens
                                    old_up = up_token_id
                                    old_down = down_token_id
                                    try:
                                        await ws.send(json.dumps({
                                            "operation": "unsubscribe",
                                            "assets_ids": [old_up, old_down],
                                        }))
                                    except Exception as unsub_err:
                                        logger.debug("Unsubscribe old tokens warning: %s", unsub_err)

                                    slot = curr_slot
                                    slug = next_slug
                                    tokens = next_tokens
                                    up_token_id = tokens["UP"]
                                    down_token_id = tokens["DOWN"]
                                    self.rollover_events_count += 1

                                    # Reset order books and invalidate until fresh book snapshots arrive
                                    up_book = LiveTokenOrderBook(up_token_id)
                                    down_book = LiveTokenOrderBook(down_token_id)
                                    self.is_book_valid = False
                                    rollover_pending_snapshot_since = time.time()

                                    # Reset feature engine so previous market's returns do not contaminate new market
                                    self.feature_engine.reset()
                                    logger.info("Feature engine and order books reset for new market %s (UP: %s, DOWN: %s)", slug, up_token_id[:10], down_token_id[:10])

                                    # Subscribe new tokens
                                    await ws.send(json.dumps({
                                        "operation": "subscribe",
                                        "assets_ids": [up_token_id, down_token_id],
                                        "custom_feature_enabled": True,
                                    }))

                                    self.rollover_records.append({
                                        "timestamp_utc": rollover_ts_utc,
                                        "from_slot": slot,
                                        "to_slot": curr_slot,
                                        "new_market": slug,
                                        "status": "SUCCESS",
                                    })
                                except Exception as ro_err:
                                    logger.warning("Rollover discovery error: %s. Will retry next cycle.", ro_err)
                                    self.rollover_records.append({
                                        "timestamp_utc": rollover_ts_utc,
                                        "from_slot": slot,
                                        "to_slot": curr_slot,
                                        "error": str(ro_err),
                                        "status": "PENDING_RETRY",
                                    })

                            # Watchdog: If rollover occurred and fresh snapshots not received within 5s, force clean reconnect
                            if rollover_pending_snapshot_since is not None and (now - rollover_pending_snapshot_since > 5.0):
                                logger.info("Fresh snapshots not received within 5s of rollover; forcing clean reconnect.")
                                rollover_pending_snapshot_since = None
                                raise ConnectionClosedOK(None, None)

                            # Drain messages from queue and update order book state
                            now_ms = int(now * 1000)
                            drain_count = 0
                            while not self.message_queue.empty() and drain_count < 500:
                                drain_count += 1
                                raw_msg = self.message_queue.get_nowait()
                                classification, parsed_data, err_detail = classify_and_parse_ws_message(raw_msg)

                                if classification == "JSON":
                                    self.json_messages_parsed += 1
                                    self._apply_parsed_market_data(parsed_data, up_book, down_book, now_ms, up_token_id, down_token_id)
                                elif classification in ("EMPTY", "CONTROL"):
                                    self.control_messages_count += 1
                                elif classification == "PROTOCOL_ERROR":
                                    self.protocol_errors_count += 1
                                    logger.warning("Protocol message received (%s): %s", slug, raw_msg)
                                    # If books uninitialized, subscription may have failed; trigger reconnect for snapshots
                                    if not (up_book.is_initialized and down_book.is_initialized):
                                        logger.info("Protocol error on uninitialized books; triggering clean reconnect.")
                                        raise ConnectionClosedOK(None, None)
                                elif classification == "MALFORMED":
                                    self.malformed_messages_count += 1
                                    snippet = repr(raw_msg[:120]) if hasattr(raw_msg, "__getitem__") else repr(raw_msg)
                                    logger.warning(
                                        "Malformed WebSocket frame received (len=%s, market=%s): %s | %s",
                                        len(raw_msg) if hasattr(raw_msg, "__len__") else 0,
                                        slug, snippet, err_detail
                                    )
                                    if len(self.malformed_samples) < 50:
                                        self.malformed_samples.append({
                                            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                            "length": len(raw_msg) if hasattr(raw_msg, "__len__") else 0,
                                            "snippet": snippet,
                                            "error": err_detail,
                                        })

                            # Check book validity: valid once both UP and DOWN have received initial snapshots
                            if up_book.is_initialized and down_book.is_initialized and not self.is_book_degraded:
                                self.is_book_valid = True
                                rollover_pending_snapshot_since = None

                            # 1-Second Grid Sampling
                            curr_sec = int(time.time())
                            if curr_sec - last_grid_sample_sec > 5:
                                # Bounded backlog clamp after reconnection
                                last_grid_sample_sec = curr_sec - 1

                            while curr_sec > last_grid_sample_sec:
                                last_grid_sample_sec += 1
                                grid_ts_ms = last_grid_sample_sec * 1000

                                if self.is_book_valid and not self.is_book_degraded:
                                    await self._process_grid_step(
                                        up_book=up_book,
                                        market_id=slug,
                                        asset_id=up_token_id,
                                        grid_timestamp_ms=grid_ts_ms,
                                    )
                                else:
                                    # Book not ready or degraded -> record uninitialized step
                                    self.uninitialized_book_steps_count += 1
                                    logger.debug("Skipping grid step %d: book uninitialized or degraded.", grid_ts_ms)

                            # Yield control briefly to event loop
                            await asyncio.sleep(0.01)

                except (ConnectionClosed, ConnectionError, OSError) as ws_err:
                    reconnect_attempts += 1
                    self.reconnect_count += 1
                    self.is_book_valid = False
                    backoff = min(base_delay * (2 ** (reconnect_attempts - 1)), max_delay) + random.uniform(0, jitter)
                    logger.warning(
                        "WebSocket disconnected (%s). Reconnecting in %.2fs (attempt %d/%d)...",
                        ws_err, backoff, reconnect_attempts, max_reconnects
                    )
                    await asyncio.sleep(backoff)
                finally:
                    if reader_task:
                        if not reader_task.done():
                            reader_task.cancel()
                        try:
                            await reader_task
                        except (asyncio.CancelledError, ConnectionClosed):
                            pass
                        except Exception as cleanup_err:
                            logger.debug("Reader task cleanup exception: %s", cleanup_err)

        except KeyboardInterrupt:
            logger.info("Session interrupted by user.")
        except Exception as fatal_err:
            self.session_fatal_error = str(fatal_err)
            logger.error("Session encountered unhandled fatal exception: %s", fatal_err, exc_info=True)
        finally:
            # Guaranteed safe finalization on ALL paths
            manifest = self._finalize_session()
            return manifest

    def _apply_parsed_market_data(
        self,
        data: Any,
        up_book: LiveTokenOrderBook,
        down_book: LiveTokenOrderBook,
        now_ms: int,
        up_token_id: str,
        down_token_id: str,
    ) -> None:
        """Apply parsed JSON events to the active token order books."""
        events = data if isinstance(data, list) else [data]
        for ev in events:
            if not isinstance(ev, dict):
                continue
            ev_type = ev.get("event_type", ev.get("type"))
            asset_id = str(ev.get("asset_id", ""))

            if ev_type == "book":
                if asset_id == up_token_id:
                    up_book.apply_book_event(ev, now_ms)
                elif asset_id == down_token_id:
                    down_book.apply_book_event(ev, now_ms)

            elif ev_type == "price_change":
                for ch in ev.get("price_changes", []):
                    ch_asset = str(ch.get("asset_id", ""))
                    if ch_asset == up_token_id:
                        up_book.apply_price_change(ch, now_ms)
                    elif ch_asset == down_token_id:
                        down_book.apply_price_change(ch, now_ms)

    async def _process_grid_step(
        self,
        up_book: LiveTokenOrderBook,
        market_id: str,
        asset_id: str,
        grid_timestamp_ms: int,
    ) -> None:
        """Execute a 1-second sampled evaluation step."""
        # 1. Check and close pending hypothetical positions whose 5s holding horizon has arrived
        for pos in list(self.pending_positions):
            if grid_timestamp_ms >= pos["target_exit_ms"]:
                exit_bid = up_book.best_bid
                exit_ask = up_book.best_ask
                has_exit_quote = (
                    exit_bid is not None
                    and exit_ask is not None
                    and exit_bid > 0
                    and exit_ask > 0
                    and exit_bid < exit_ask
                )
                if has_exit_quote:
                    pos["has_exit_quote"] = True
                    pos["exit_timestamp_ms"] = grid_timestamp_ms
                    pos["bid_exit"] = exit_bid
                    pos["ask_exit"] = exit_ask
                else:
                    pos["has_exit_quote"] = False
                    pos["exit_timestamp_ms"] = None
                    pos["bid_exit"] = None
                    pos["ask_exit"] = None
                self._log_and_complete_position(pos)
                self.pending_positions.remove(pos)

        # 2. Extract top-of-book quotes for decision evaluation
        bid = up_book.best_bid
        ask = up_book.best_ask
        bid_size = up_book.bid_size
        ask_size = up_book.ask_size

        if bid is None or ask is None or bid <= 0 or ask <= 0 or bid >= ask:
            self.unquotable_quotes_count += 1
            return

        quote_age_ms = float(grid_timestamp_ms - up_book.last_update_ms) if up_book.last_update_ms > 0 else 0.0
        data_gap_ms = float(grid_timestamp_ms - self.last_event_ms) if self.last_event_ms > 0 else 1000.0
        if data_gap_ms > 5000.0:
            self.burst_id += 1
        self.last_event_ms = grid_timestamp_ms

        if quote_age_ms > 2000.0:
            self.stale_quotes_count += 1

        # 3. Add quote to online feature engine
        self.feature_engine.add_grid_sample(
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp_ms=grid_timestamp_ms,
        )

        # 4. If warmed up, perform inference and gating
        feat_res = self.feature_engine.compute_current_features_and_sequence()
        if feat_res is None:
            self.warmup_steps_count += 1
            return  # Still warming up initial 10 seconds

        curr_raw_feats, scaled_tensor = feat_res
        scaled_spread = float(scaled_tensor[0, 9, 1])
        scaled_depth_imbal = float(scaled_tensor[0, 9, 10])
        abs_scaled_depth_imbal = abs(scaled_depth_imbal)

        # Model inference
        with torch.no_grad():
            x_torch = torch.from_numpy(scaled_tensor)
            logits = self.model(x_torch)
            probs = torch.softmax(logits, dim=-1).numpy()[0]

        prob_down, prob_flat, prob_up = float(probs[0]), float(probs[1]), float(probs[2])
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        # Evaluate gating (new entries strictly halted if hard stop triggered)
        is_directional = pred_class in (0, 2)
        gate_passed = (
            not self.is_hard_stopped
            and is_directional
            and (confidence >= PREDECLARED_CONFIDENCE_THRESHOLD)
            and (scaled_spread <= TRAIN_SPREAD_THRESHOLD)
            and (abs_scaled_depth_imbal <= TRAIN_DEPTH_IMBALANCE_P90)
            and (quote_age_ms <= 2000.0)
            and (data_gap_ms <= 5000.0)
        )

        self.observations_logged += 1

        if not gate_passed:
            if pred_class == 1:
                self.trades_skipped_flat += 1
            else:
                self.trades_suppressed += 1

            self.shadow_logger.log_decision_point(
                event_timestamp_ms=grid_timestamp_ms,
                market_id=market_id,
                asset_id=asset_id,
                quote_timestamp_ms=up_book.last_update_ms or grid_timestamp_ms,
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
                exit_timestamp_ms=None,
                bid_exit=None,
                ask_exit=None,
                hard_stop_active=self.is_hard_stopped,
            )
        else:
            self.trades_executed += 1
            pos_record = {
                "event_timestamp_ms": grid_timestamp_ms,
                "market_id": market_id,
                "asset_id": asset_id,
                "quote_timestamp_ms": up_book.last_update_ms or grid_timestamp_ms,
                "bid_entry": bid,
                "ask_entry": ask,
                "depth_imbalance": float(curr_raw_feats[10]),
                "scaled_spread": scaled_spread,
                "abs_scaled_depth_imbal": abs_scaled_depth_imbal,
                "model_probs": (prob_down, prob_flat, prob_up),
                "predicted_class": pred_class,
                "confidence": confidence,
                "burst_id": self.burst_id,
                "data_gap_ms": data_gap_ms,
                "target_exit_ms": grid_timestamp_ms + 5000,
                "bid_exit": None,
                "ask_exit": None,
                "exit_timestamp_ms": None,
            }
            self.pending_positions.append(pos_record)

    def _log_and_complete_position(self, pos: Dict[str, Any]) -> None:
        """Write completed hypothetical trade record to append-only log and update hard-stop risk state."""
        record = self.shadow_logger.log_decision_point(
            event_timestamp_ms=pos["event_timestamp_ms"],
            market_id=pos["market_id"],
            asset_id=pos["asset_id"],
            quote_timestamp_ms=pos["quote_timestamp_ms"],
            bid_entry=pos["bid_entry"],
            ask_entry=pos["ask_entry"],
            depth_imbalance=pos["depth_imbalance"],
            scaled_spread=pos["scaled_spread"],
            abs_scaled_depth_imbal=pos["abs_scaled_depth_imbal"],
            model_probs=pos["model_probs"],
            predicted_class=pos["predicted_class"],
            confidence=pos["confidence"],
            burst_id=pos["burst_id"],
            data_gap_ms=pos["data_gap_ms"],
            exit_timestamp_ms=pos["exit_timestamp_ms"],
            bid_exit=pos["bid_exit"],
            ask_exit=pos["ask_exit"],
            fee_bps=5.0,
            adverse_slippage=0.0,
        )

        # Update hard-stop metrics on completed trade
        if record.get("outcome_status") == "COMPLETED":
            net_pnl = record.get("net_hypothetical_pnl")
            if net_pnl is not None:
                if net_pnl < -1e-6:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

                self.cum_net_pnl += net_pnl
                if self.cum_net_pnl > self.peak_net_pnl:
                    self.peak_net_pnl = self.cum_net_pnl
                current_dd = self.peak_net_pnl - self.cum_net_pnl
                if current_dd > self.max_drawdown_seen:
                    self.max_drawdown_seen = current_dd

                # Check predeclared hard-stopping triggers immediately on trade completion
                if not self.is_hard_stopped:
                    if self.consecutive_losses >= self.consecutive_losses_limit:
                        self.is_hard_stopped = True
                        self.hard_stop_reason = f"CONSECUTIVE_LOSSES_LIMIT ({self.consecutive_losses} >= {self.consecutive_losses_limit})"
                        logger.warning(
                            "Hard stopping abort trigger activated: %s. Halting new entries immediately.",
                            self.hard_stop_reason
                        )
                    elif current_dd >= self.max_drawdown_stop_loss_units:
                        self.is_hard_stopped = True
                        self.hard_stop_reason = f"MAX_DRAWDOWN_LIMIT ({current_dd:.4f} >= {self.max_drawdown_stop_loss_units})"
                        logger.warning(
                            "Hard stopping abort trigger activated: %s. Halting new entries immediately.",
                            self.hard_stop_reason
                        )

    def _finalize_session(self) -> Dict[str, Any]:
        """
        Guarantee safe finalization:
        - Flushes pending positions as INCOMPLETE_MISSING_EXIT.
        - Closes logger and computes integrity hash.
        - Verifies model weight immutability.
        - Classifies session status: COMPLETE, DEGRADED, or FAILED.
        - Writes manifest and health report.
        """
        for pos in self.pending_positions:
            self.shadow_logger.log_decision_point(
                event_timestamp_ms=pos["event_timestamp_ms"],
                market_id=pos["market_id"],
                asset_id=pos["asset_id"],
                quote_timestamp_ms=pos["quote_timestamp_ms"],
                bid_entry=pos["bid_entry"],
                ask_entry=pos["ask_entry"],
                depth_imbalance=pos["depth_imbalance"],
                scaled_spread=pos["scaled_spread"],
                abs_scaled_depth_imbal=pos["abs_scaled_depth_imbal"],
                model_probs=pos["model_probs"],
                predicted_class=pos["predicted_class"],
                confidence=pos["confidence"],
                burst_id=pos["burst_id"],
                data_gap_ms=pos["data_gap_ms"],
                exit_timestamp_ms=None,
                bid_exit=None,
                ask_exit=None,
            )
        self.pending_positions.clear()

        # Close append-only logger
        logger_manifest = self.shadow_logger.close()

        # Verify model checkpoint immutability
        final_model_hash = sha256_file(self.model_path)
        if self.model_initial_hash != final_model_hash:
            raise RuntimeError("CRITICAL ERROR: Model weights mutated during prospective shadow session!")

        end_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Accurate status classification
        if self.session_fatal_error is not None:
            session_status = "FAILED"
        elif self.hard_stop_reason is not None:
            session_status = "ABORTED_HARD_STOP"
        elif self.queue_overflow_count > 0 or self.malformed_messages_count > 0 or self.reconnect_count >= 3:
            session_status = "DEGRADED"
        else:
            session_status = "COMPLETE"

        # Generate health report
        health_report_path = self.session_dir / "collector_health_report.md"
        self._write_health_report(health_report_path, logger_manifest, end_utc, session_status)

        # Generate session provenance manifest
        manifest_path = self.session_dir / "session_manifest.json"
        session_manifest = {
            "session_id": self.session_id,
            "session_type": "GENUINE_PROSPECTIVE_SHADOW",
            "session_status": session_status,
            "hard_stop_triggered": self.hard_stop_reason,
            "start_time_utc": self.start_utc,
            "end_time_utc": end_utc,
            "total_observations_logged": self.observations_logged,
            "warmup_steps_count": self.warmup_steps_count,
            "unquotable_quotes_count": self.unquotable_quotes_count,
            "uninitialized_book_steps_count": self.uninitialized_book_steps_count,
            "trades_executed": self.trades_executed,
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
            "health_report_sha256": sha256_file(health_report_path),
            "locked_test_set_touched": False,
            "real_capital_used": False,
            "exchange_orders_sent": False,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(session_manifest, f, indent=2)

        logger.info(
            "Session %s finalized [%s]. Output saved to %s",
            self.session_id, session_status, self.session_dir
        )
        return session_manifest

    def _write_health_report(
        self,
        path: Path,
        logger_manifest: Dict[str, Any],
        end_utc: str,
        session_status: str,
    ) -> None:
        """Write collector health and telemetry markdown report."""
        status_badge = (
            "**HEALTHY (COMPLETE)**" if session_status == "COMPLETE"
            else ("**ABORTED (HARD STOP TRIGGERED)**" if session_status == "ABORTED_HARD_STOP"
            else ("**DEGRADED**" if session_status == "DEGRADED" else "**FAILED**"))
        )

        md = [
            f"# Prospective Shadow Collector Health Report — Session `{self.session_id}`\n",
            "> [!IMPORTANT]",
            "> **RESEARCH-ONLY PROSPECTIVE TELEMETRY**: Zero real capital deployed; zero live orders sent to exchange.",
            "> All hypothetical returns represent simulated paper executions under conservative bid/ask crossing and 5 bps taker fees.\n",
            "---\n",
            f"## 1. Session Status: {status_badge}\n",
            "| Telemetry Metric | Session Value | Quality Benchmark |",
            "| :--- | :---: | :--- |",
            f"| **Start Timestamp (UTC)** | `{self.start_utc}` | Recorded |",
            f"| **End Timestamp (UTC)** | `{end_utc}` | Recorded |",
            f"| **Session Status** | `{session_status}` | Operational certification |",
            f"| **Raw WebSocket Frames** | `{self.ws_messages_received:,}` | Active feed connectivity |",
            f"| **JSON Payloads Parsed** | `{self.json_messages_parsed:,}` | Normal market data |",
            f"| **Heartbeat / Control Frames** | `{self.control_messages_count:,}` | PING/PONG/OK handled gracefully |",
            f"| **Protocol Status / Rejections** | `{self.protocol_errors_count:,}` | 0 expected (handled without crash) |",
            f"| **Malformed / Non-JSON Frames** | `{self.malformed_messages_count:,}` | 0 expected (logged safely) |",
            f"| **Queue Overflows (Dropped)** | `{self.queue_overflow_count:,}` | 0 required for strict continuity |",
            f"| **Peak Ingestion Queue Depth** | `{self.max_queue_depth_seen:,}` | Capacity: {self.queue_maxsize:,} |",
            f"| **Reconnect Attempts** | `{self.reconnect_count:,}` | Resubscription & snapshot rebuild |",
            f"| **Market Rollovers** | `{self.rollover_events_count:,}` | Feature engine cleanly reset |",
            f"| **1-Second Decisions Logged** | `{self.observations_logged:,}` | Grid synchronization |",
            f"| **Feature Warm-up Steps** | `{self.warmup_steps_count:,}` | Initial sequence building |",
            f"| **Unquotable / Crossed Quotes** | `{self.unquotable_quotes_count:,}` | 0 valid opportunities during unquotable book |",
            f"| **Uninitialized Book Steps** | `{self.uninitialized_book_steps_count:,}` | Steps waiting for initial snapshots |",
            f"| **Trades Triggered (EXECUTE)** | `{self.trades_executed:,}` | Evaluated under frozen regime gates |",
            f"| **Trades Suppressed by Gates** | `{self.trades_suppressed:,}` | Strict regime gating |",
            f"| **Predictions Skipped (FLAT)** | `{self.trades_skipped_flat:,}` | Non-directional model output |",
            f"| **Stale Quote Events** | `{self.stale_quotes_count:,}` | Quote age > 2000ms threshold |\n",
            "---\n",
            "## 2. Invariants & Safety Verification\n",
            "- [x] **Zero Live Orders**: Process operates strictly as a read-only WebSocket subscriber.",
            "- [x] **Model Weights Frozen**: SmallLSTM weights remained bitwise identical (SHA-256 verified).",
            "- [x] **Locked Test Isolation**: `test_scaled.npz` was never accessed or opened.",
            "- [x] **Append-Only Logging**: Every observation hashed with SHA-256 rolling digest.",
            f"- [x] **Observation Log File**: `{logger_manifest['log_file']}` (SHA-256: `{logger_manifest['log_sha256'][:16]}...`).\n",
        ]

        if self.hard_stop_reason:
            md.extend([
                "---\n",
                "## 3. Hard Stopping Abort Trigger\n",
                f"- **Trigger Code**: `{self.hard_stop_reason}`\n",
                "- **Enforcement**: New hypothetical trade entries halted immediately; pending in-flight positions drained before session exit.\n",
            ])

        if self.session_fatal_error:
            md.extend([
                "---\n",
                "## 3. Fatal Error Diagnostics\n",
                f"```text\n{self.session_fatal_error}\n```\n",
            ])

        if self.malformed_samples:
            md.extend([
                "---\n",
                "## 4. Malformed Message Samples (First 5)\n",
                "| Timestamp UTC | Length | Sample Snippet | Error Detail |",
                "| :--- | :---: | :--- | :--- |",
            ])
            for smp in self.malformed_samples[:5]:
                md.append(f"| `{smp['timestamp_utc']}` | {smp['length']} | `{smp['snippet']}` | `{smp['error']}` |")
            md.append("\n")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")


# -----------------------------------------------------------------------------
# Section 5: CLI Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Start Genuine Prospective Shadow Session for PredAlpha-HFT")
    parser.add_argument("--duration-seconds", type=int, default=None, help="Stop session after N seconds (e.g. 60 or 300)")
    parser.add_argument("--max-observations", type=int, default=None, help="Stop session after N grid observations")
    parser.add_argument("--session-id", type=str, default=None, help="Custom session ID string")
    parser.add_argument("--out-dir", type=Path, default=PROSPECTIVE_BASE_DIR, help="Base directory for prospective output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    session = ProspectiveShadowSession(
        output_base_dir=args.out_dir,
        session_id=args.session_id,
        duration_seconds=args.duration_seconds,
        max_observations=args.max_observations,
    )

    manifest = asyncio.run(session.run())

    print("\n" + "=" * 60)
    print(f"PROSPECTIVE SHADOW SESSION FINALIZED [{manifest['session_status']}]")
    print(f"Session ID:         {manifest['session_id']}")
    print(f"Decisions Logged:   {manifest['total_observations_logged']}")
    print(f"Trades Executed:    {manifest['trades_executed']}")
    print(f"Trades Suppressed:  {manifest['trades_suppressed']}")
    print(f"WS Frames Ingested: {manifest['ws_messages_received']}")
    print(f"Queue Peak Depth:   {manifest['max_queue_depth_seen']}")
    print(f"Log File:           {manifest['log_file']}")
    print(f"SHA-256:            {manifest['log_sha256']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
