"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { ProbabilityBar } from "@/components/ProbabilityBar";
import { ExpandableSection } from "@/components/ExpandableSection";
import { runPrediction, PredictResponse } from "@/lib/api-client";

interface ScenarioItem {
  id: string;
  name: string;
  tag: string;
  desc: string;
}

const SCENARIOS: ScenarioItem[] = [
  {
    id: "bullish",
    name: "Heavy Bid Depth Wall",
    tag: "Upward Bias",
    desc: "Buyer-dominated touch liquidity (+0.65 imbalance), rising returns, positive quote velocity.",
  },
  {
    id: "bearish",
    name: "Heavy Ask Depth Wall",
    tag: "Downward Bias",
    desc: "Persistent ask overhang (-0.65 imbalance), downward return drift, negative quote velocity.",
  },
  {
    id: "stationary",
    name: "Stationary Spread Regime",
    tag: "Neutral Drift",
    desc: "Balanced touch queue (~0.0 imbalance), negligible drift, tight 1-tick spread.",
  },
  {
    id: "volatile",
    name: "Volatility Spread Shock",
    tag: "Inversion Risk",
    desc: "Widened spread (550 bps), elevated 5s realized volatility, unstable touch queues.",
  },
  {
    id: "custom",
    name: "Custom Parameter Matrix",
    tag: "User Defined",
    desc: "Interactive parameter modulation applied directly to the terminal snapshot.",
  },
];

const INITIAL_BULLISH_RESULT: PredictResponse = {
  scenario: "bullish",
  scenario_description:
    "Bullish Order Flow: persistent positive depth imbalance (+0.65), rising mid returns, upward quote momentum.",
  prediction: {
    predicted_label: "Up",
    predicted_symbol: "Up",
    confidence: 0.686,
    temperature: 0.85,
    probabilities: {
      "Down (0)": 0.205,
      "Stationary (1)": 0.109,
      "Up (2)": 0.686,
    },
    raw_logits: [-0.12, -0.66, 0.907],
    entropy: 0.824,
    uncertainty_pct: 75.0,
  },
  feature_names: [
    "mid_price",
    "spread",
    "spread_bps",
    "microprice",
    "depth_imbalance",
    "mid_return_1s",
    "mid_return_3s",
    "mid_return_5s",
    "mid_volatility_5s",
    "bid_change_1s",
    "ask_change_1s",
  ],
  scaled_sequence: [],
  raw_sequence: [
    [50000.0, 0.5, 1.0, 50000.1, 0.35, 0.0001, 0.0003, 0.0005, 0.0002, 0.1, 0.05],
    [50000.1, 0.5, 1.0, 50000.2, 0.40, 0.0002, 0.0004, 0.0006, 0.0002, 0.1, 0.05],
    [50000.2, 0.5, 1.0, 50000.3, 0.45, 0.0002, 0.0005, 0.0007, 0.0002, 0.15, 0.05],
    [50000.3, 0.5, 1.0, 50000.4, 0.50, 0.0003, 0.0006, 0.0008, 0.0002, 0.15, 0.05],
    [50000.4, 0.5, 1.0, 50000.5, 0.55, 0.0003, 0.0007, 0.0009, 0.0002, 0.2, 0.05],
    [50000.5, 0.5, 1.0, 50000.6, 0.58, 0.0004, 0.0008, 0.001, 0.0003, 0.2, 0.05],
    [50000.6, 0.5, 1.0, 50000.7, 0.60, 0.0004, 0.0009, 0.0011, 0.0003, 0.25, 0.05],
    [50000.7, 0.5, 1.0, 50000.8, 0.62, 0.0005, 0.001, 0.0012, 0.0003, 0.25, 0.05],
    [50000.8, 0.5, 1.0, 50000.9, 0.64, 0.0005, 0.0011, 0.0013, 0.0003, 0.3, 0.05],
    [50000.9, 0.5, 1.0, 50001.0, 0.65, 0.0006, 0.0012, 0.0015, 0.0003, 0.3, 0.05],
  ],
  engine_source: "Live PyTorch Runtime (best_lstm.pt, CPU)",
};

export default function ModelLabPage() {
  const [selectedScenario, setSelectedScenario] = useState("bullish");
  const [temperature, setTemperature] = useState(0.85);
  const [depthImbalance, setDepthImbalance] = useState(0.50);
  const [spreadBps, setSpreadBps] = useState(180.0);
  const [midReturn5s, setMidReturn5s] = useState(0.005);
  const [volatility5s, setVolatility5s] = useState(0.07);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(INITIAL_BULLISH_RESULT);
  const [error, setError] = useState<string | null>(null);

  const cacheRef = useRef<Map<string, PredictResponse>>(
    new Map([["bullish_0.85", INITIAL_BULLISH_RESULT]])
  );
  const abortControllerRef = useRef<AbortController | null>(null);
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  const fetchInference = useCallback(
    async (
      scenario: string,
      temp: number,
      overrides?: Record<string, number>
    ) => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;

      setLoading(true);
      setError(null);

      try {
        const data = await runPrediction(scenario, temp, overrides, controller.signal);
        setResult(data);
        const cacheKey =
          scenario === "custom"
            ? `custom_${temp}_${overrides?.depth_imbalance}_${overrides?.spread_bps}_${overrides?.mid_return_5s}_${overrides?.mid_volatility_5s}`
            : `${scenario}_${temp}`;
        cacheRef.current.set(cacheKey, data);
      } catch (err: any) {
        if (err.name === "AbortError" || err.message?.includes("aborted")) {
          return;
        }
        setError(err.message || "Failed to execute inference");
      } finally {
        if (abortControllerRef.current === controller) {
          setLoading(false);
        }
      }
    },
    []
  );

  useEffect(() => {
    const overrides =
      selectedScenario === "custom"
        ? {
            depth_imbalance: depthImbalance,
            spread_bps: spreadBps,
            mid_return_5s: midReturn5s,
            mid_volatility_5s: volatility5s,
          }
        : undefined;

    const cacheKey =
      selectedScenario === "custom"
        ? `custom_${temperature}_${depthImbalance}_${spreadBps}_${midReturn5s}_${volatility5s}`
        : `${selectedScenario}_${temperature}`;

    // Instant optimistic cache check
    const cached = cacheRef.current.get(cacheKey);
    if (cached) {
      setResult(cached);
      setLoading(false);
      return;
    }

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // Immediate execution for preset buttons, 100ms debounce for continuous sliders
    const delay = selectedScenario === "custom" ? 100 : 0;
    debounceTimerRef.current = setTimeout(() => {
      fetchInference(selectedScenario, temperature, overrides);
    }, delay);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [selectedScenario, temperature, depthImbalance, spreadBps, midReturn5s, volatility5s, fetchInference]);

  // Operational threshold check (tau >= 0.65 from empirical validation findings)
  const isGated = result ? result.prediction.confidence < 0.65 : false;

  // Active touch queue parameters for Level-2 depth visualization
  const activeImbalance =
    selectedScenario === "custom"
      ? depthImbalance
      : selectedScenario === "bullish"
      ? 0.65
      : selectedScenario === "bearish"
      ? -0.65
      : selectedScenario === "volatile"
      ? -0.15
      : 0.0;

  const activeSpreadBps =
    selectedScenario === "custom"
      ? Math.round(spreadBps)
      : selectedScenario === "volatile"
      ? 550
      : selectedScenario === "stationary"
      ? 100
      : 200;

  const bidRatio = Math.max(5, Math.min(95, Math.round(((1 + activeImbalance) / 2) * 100)));
  const askRatio = 100 - bidRatio;

  return (
    <article className="space-y-10">
      {/* 1. Workbench Header — Clean, open analytical workspace layout */}
      <header className="space-y-4 pb-6 border-b border-paper-rule">
        <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-ink-muted">
          <div className="flex items-center space-x-3">
            <span className="font-semibold text-ink-primary uppercase tracking-wide text-[11px]">
              Microstructure Inference Workbench
            </span>
            <span className="text-paper-rule">/</span>
            <span className="text-brass font-medium text-[11px]">
              Genuine PyTorch CPU Forward Pass
            </span>
          </div>

          <div className="flex items-center space-x-3 text-[11px] text-ink-faint">
            <span>Model: best_lstm.pt (<span className="numeric-value">5,859</span> weights)</span>
            <span>·</span>
            <span>Horizon: <span className="numeric-value">5,000 ms</span></span>
          </div>
        </div>

        <div className="space-y-2 max-w-3xl">
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight text-ink-primary">
            Level-2 Microstructure Inference Workbench
          </h1>
          <p className="text-xs sm:text-sm text-ink-body leading-relaxed">
            Evaluate how touch queue imbalances, quotation velocity, and historical return memory map into 
            forward 5-second directional probabilities. Every query performs an authentic forward pass through 
            the SmallLSTM PyTorch network using historical feature standardization parameters.
          </p>
        </div>

        {/* Scientific Context Bar: Clear demarcation of synthetic vs historical data */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-3 rounded-lg border border-paper-rule bg-paper-canvas/50 text-xs">
          <div className="flex items-center space-x-2 text-ink-muted">
            <span className="font-medium text-ink-primary">Scientific Notice:</span>
            <span>This workbench evaluates synthetic scenario profiles to study model response. Out-of-sample generalization metrics are established on the 1,428-sequence historical holdout.</span>
          </div>
          <span className="text-[11px] font-mono text-ink-faint shrink-0">
            Phase 19 Schema
          </span>
        </div>
      </header>

      {/* 2. Analytical Workspace: 2-Column Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Regime Navigator & Control Surface (5 cols) */}
        <section aria-label="Input Controls" className="lg:col-span-5 space-y-6">
          {/* Regime Selector */}
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold uppercase tracking-wide text-[11px] text-ink-muted">
                Order Book Regime Scenario
              </span>
              <span className="text-[11px] text-ink-faint">5 Evaluation Regimes</span>
            </div>

            <div className="border border-paper-rule rounded-xl bg-paper-canvas/40 divide-y divide-paper-rule/60 overflow-hidden">
              {SCENARIOS.map((sc) => {
                const isSelected = selectedScenario === sc.id;
                return (
                  <button
                    key={sc.id}
                    onClick={() => setSelectedScenario(sc.id)}
                    className={`w-full text-left p-3.5 transition-colors ${
                      isSelected
                        ? "bg-paper-surface text-ink-primary"
                        : "hover:bg-paper-surface/40 text-ink-muted hover:text-ink-body"
                    }`}
                  >
                    <div className="flex items-baseline justify-between mb-0.5">
                      <span className={`text-xs font-semibold ${isSelected ? "text-ink-primary" : "text-ink-body"}`}>
                        {sc.name}
                      </span>
                      <span className={`text-[10px] font-mono ${
                        isSelected ? "text-brass font-medium" : "text-ink-faint"
                      }`}>
                        {sc.tag}
                      </span>
                    </div>
                    <p className="text-[11px] text-ink-muted leading-relaxed">
                      {sc.desc}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Temperature Scaling Control */}
          <div className="border border-paper-rule rounded-xl bg-paper-canvas/40 p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-ink-primary">
                Softmax Temperature Calibration
              </span>
              <span className="px-2 py-0.5 rounded bg-paper-surface border border-paper-rule text-brass text-xs font-semibold numeric-value">
                T = {temperature.toFixed(2)}
              </span>
            </div>

            <input
              type="range"
              min="0.30"
              max="2.00"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              aria-label="Softmax temperature calibration"
              className="w-full accent-brass bg-paper-rule h-1.5 rounded-full cursor-pointer focus:outline-none"
            />

            <div className="flex justify-between text-[10px] text-ink-faint px-0.5">
              <span><span className="numeric-value">0.30</span> (Argmax)</span>
              <span className="text-ink-primary font-medium"><span className="numeric-value">0.85</span> (Calibrated)</span>
              <span><span className="numeric-value">2.00</span> (Soft)</span>
            </div>

            <p className="text-[11px] text-ink-muted leading-relaxed pt-2 border-t border-paper-rule/60">
              Temperature scaling calibrates prediction margins against empirical frequencies (<span className="font-mono text-ink-body">z_i / T</span>) without altering the argmax directional decision.
            </p>
          </div>

          {/* Custom Feature Modulation Panel (only when custom scenario selected) */}
          {selectedScenario === "custom" && (
            <div className="border border-paper-rule rounded-xl bg-paper-canvas/50 p-4 space-y-4">
              <div className="flex items-baseline justify-between border-b border-paper-rule/60 pb-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-ink-primary text-[11px]">
                  Terminal Feature Modulation (t0)
                </span>
                <span className="text-[10px] font-mono text-brass">Active</span>
              </div>

              <div className="space-y-3.5 text-xs">
                {/* Imbalance */}
                <div className="space-y-1">
                  <div className="flex justify-between text-ink-body">
                    <span>Touch Queue Imbalance:</span>
                    <span className="font-mono font-semibold text-ink-primary tabular-nums">
                      {depthImbalance > 0 ? `+${depthImbalance.toFixed(2)}` : depthImbalance.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="-1.0"
                    max="1.0"
                    step="0.05"
                    value={depthImbalance}
                    onChange={(e) => setDepthImbalance(parseFloat(e.target.value))}
                    className="w-full accent-brass bg-paper-rule h-1.5 rounded-full cursor-pointer"
                  />
                  <div className="flex justify-between font-mono text-[9px] text-ink-faint">
                    <span>-1.0 (Full Ask)</span>
                    <span>0.0 (Neutral)</span>
                    <span>+1.0 (Full Bid)</span>
                  </div>
                </div>

                {/* Spread */}
                <div className="space-y-1">
                  <div className="flex justify-between text-ink-body">
                    <span>Spread Width:</span>
                    <span className="font-mono font-semibold text-ink-primary tabular-nums">
                      {spreadBps.toFixed(0)} bps
                    </span>
                  </div>
                  <input
                    type="range"
                    min="50"
                    max="600"
                    step="10"
                    value={spreadBps}
                    onChange={(e) => setSpreadBps(parseFloat(e.target.value))}
                    className="w-full accent-brass bg-paper-rule h-1.5 rounded-full cursor-pointer"
                  />
                </div>

                {/* 5s Return */}
                <div className="space-y-1">
                  <div className="flex justify-between text-ink-body">
                    <span>5s Mid Return:</span>
                    <span className="font-mono font-semibold text-ink-primary tabular-nums">
                      {(midReturn5s * 100).toFixed(2)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="-0.02"
                    max="0.02"
                    step="0.001"
                    value={midReturn5s}
                    onChange={(e) => setMidReturn5s(parseFloat(e.target.value))}
                    className="w-full accent-brass bg-paper-rule h-1.5 rounded-full cursor-pointer"
                  />
                </div>

                {/* 5s Volatility */}
                <div className="space-y-1">
                  <div className="flex justify-between text-ink-body">
                    <span>5s Realized Volatility:</span>
                    <span className="font-mono font-semibold text-ink-primary tabular-nums">
                      {(volatility5s * 100).toFixed(1)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.01"
                    max="0.30"
                    step="0.01"
                    value={volatility5s}
                    onChange={(e) => setVolatility5s(parseFloat(e.target.value))}
                    className="w-full accent-brass bg-paper-rule h-1.5 rounded-full cursor-pointer"
                  />
                </div>
              </div>
            </div>
          )}

          <div className="text-[11px] text-ink-faint leading-relaxed px-1">
            * Ingested features are normalized via frozen scaler parameters (μ, σ) fit exclusively on the historical training partition.
          </div>
        </section>

        {/* Right Column: Model Inference Outputs & Diagnostic Instruments (7 cols) */}
        <section aria-label="Inference Outputs" className="lg:col-span-7 space-y-6">
          {error && (
            <div className="rounded-lg border border-direction-down/40 bg-direction-down/5 p-4 text-xs text-direction-down">
              <strong>Inference Fault:</strong> {error}
            </div>
          )}

          {result && (
            <div className="space-y-6">
              {/* Primary Direction & Confidence Instrument */}
              <div className="border border-paper-rule rounded-xl bg-paper-canvas/50 p-6 space-y-5">
                <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-paper-rule/60 pb-3">
                  <div className="text-[11px] uppercase tracking-wide font-semibold text-ink-muted">
                    Model Inference Diagnostic · 5,000 ms Horizon
                  </div>
                  <div className="text-[11px] font-mono text-ink-faint">
                    {result.engine_source || "PyTorch Engine"}
                  </div>
                </div>

                <div className="flex flex-wrap items-baseline justify-between gap-4">
                  <div className="space-y-1">
                    <span className="text-[11px] text-ink-faint uppercase tracking-wide block">Forecast Outcome</span>
                    <div className="text-3xl sm:text-4xl font-bold tracking-tight">
                      {result.prediction.predicted_label === "Up" && (
                        <span className="text-direction-up">Class 2 · Up</span>
                      )}
                      {result.prediction.predicted_label === "Down" && (
                        <span className="text-direction-down">Class 0 · Down</span>
                      )}
                      {result.prediction.predicted_label === "Stationary" && (
                        <span className="text-ink-primary">Class 1 · Stationary</span>
                      )}
                    </div>
                  </div>

                  <div className="space-y-1 text-right">
                    <span className="text-[11px] text-ink-faint uppercase tracking-wide block">Softmax Probability</span>
                    <div className="text-3xl sm:text-4xl font-bold tracking-tight text-ink-primary numeric-value">
                      {(result.prediction.confidence * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="w-full pt-2 flex items-center justify-between text-xs border-t border-paper-rule/60">
                    <span className="text-ink-muted">Execution Gate (Threshold τ ≥ <span className="numeric-value">0.65</span>):</span>
                    {isGated ? (
                      <span className="text-direction-down font-medium">
                        Filtered (τ &lt; <span className="numeric-value">0.65</span>) — Sub-threshold conviction
                      </span>
                    ) : (
                      <span className="text-direction-up font-medium">
                        Eligible (τ ≥ <span className="numeric-value">0.65</span>) — High-conviction signal
                      </span>
                    )}
                  </div>
                </div>

                {/* Level-2 Order-Book Touch Depth Gauge */}
                <div className="space-y-2 pt-2 border-t border-paper-rule/60">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-ink-muted font-medium text-[11px]">
                      Touch Queue Depth Profile (Spread: <span className="numeric-value">{activeSpreadBps} bps</span>)
                    </span>
                    <span className="text-ink-faint text-[11px]">
                      Imbalance: <strong className={(activeImbalance > 0 ? "text-direction-up" : activeImbalance < 0 ? "text-direction-down" : "text-ink-primary") + " numeric-value"}>
                        {activeImbalance > 0 ? `+${activeImbalance.toFixed(2)}` : activeImbalance.toFixed(2)}
                      </strong>
                    </span>
                  </div>

                  <div className="flex h-2 w-full overflow-hidden bg-paper-canvas rounded-full border border-paper-rule">
                    <div
                      className="bg-direction-up transition-[width] duration-300 ease-out"
                      style={{ width: `${bidRatio}%` }}
                      title={`Bid Depth: ${bidRatio}%`}
                    />
                    <div
                      className="bg-direction-down transition-[width] duration-300 ease-out"
                      style={{ width: `${askRatio}%` }}
                      title={`Ask Depth: ${askRatio}%`}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-ink-faint">
                    <span>Q_bid: <span className="numeric-value">{bidRatio}%</span></span>
                    <span>Q_ask: <span className="numeric-value">{askRatio}%</span></span>
                  </div>
                </div>

                <p className="text-xs text-ink-muted leading-relaxed pt-1">
                  {result.scenario_description}
                </p>
              </div>

              {/* Ternary Probability Distribution Bar */}
              <div className="border border-paper-rule rounded-xl bg-paper-canvas/40 p-5 space-y-4">
                <div className="flex items-baseline justify-between border-b border-paper-rule/60 pb-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-muted text-[11px]">
                    Ternary Probability Vector P(Δmid)
                  </h3>
                  <span className="text-[11px] text-ink-faint">
                    Σ P(ci) = <span className="numeric-value">1.000</span>
                  </span>
                </div>

                <ProbabilityBar
                  probabilities={result.prediction.probabilities}
                  predictedLabel={result.prediction.predicted_label}
                  logits={result.prediction.raw_logits}
                />
              </div>

              {/* Information-Theoretic Diagnostics */}
              <div className="grid grid-cols-2 gap-4">
                <div className="border border-paper-rule rounded-xl bg-paper-canvas/40 p-4 space-y-1">
                  <div className="text-[11px] text-ink-faint uppercase tracking-wide">Shannon Entropy H(P)</div>
                  <div className="text-2xl font-bold text-ink-primary numeric-value">
                    {result.prediction.entropy.toFixed(3)} <span className="text-xs text-ink-muted font-normal">nats</span>
                  </div>
                  <p className="text-[11px] text-ink-muted leading-normal">
                    Max entropy = <span className="numeric-value">1.099</span> nats (uniform dispersion).
                  </p>
                </div>

                <div className="border border-paper-rule rounded-xl bg-paper-canvas/40 p-4 space-y-1">
                  <div className="text-[11px] text-ink-faint uppercase tracking-wide">Normalized Uncertainty</div>
                  <div className="text-2xl font-bold text-ink-primary numeric-value">
                    {result.prediction.uncertainty_pct.toFixed(1)}%
                  </div>
                  <p className="text-[11px] text-ink-muted leading-normal">
                    Dispersion relative to maximal entropy state.
                  </p>
                </div>
              </div>

              {/* Causal Input Sequence Tensor Table */}
              <ExpandableSection
                title="Causal Input Sequence Tensor [1, 10, 11]"
                summary="Inspect the 10 sequential 1-second snapshots unrolled into the LSTM cell"
                badge="Data Matrix"
              >
                <div className="overflow-x-auto border border-paper-rule rounded-lg bg-paper-surface/40 p-3 mt-2">
                  <table className="w-full text-left font-mono text-[10px]">
                    <thead>
                      <tr className="border-b border-paper-rule text-ink-faint font-sans">
                        <th className="py-1 px-2 whitespace-nowrap">Step</th>
                        {result.feature_names.map((fn) => (
                          <th key={fn} className="py-1 px-2 whitespace-nowrap">
                            {fn}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-paper-rule/60 text-ink-muted">
                      {result.raw_sequence.map((step, idx) => (
                        <tr key={idx} className="hover:bg-paper-surface">
                          <td className="py-1 px-2 font-semibold text-ink-primary whitespace-nowrap">
                            {idx < 9 ? `t-${9 - idx}s` : "t0 (touch)"}
                          </td>
                          {step.map((val, cIdx) => (
                            <td key={cIdx} className="py-1 px-2 tabular-nums whitespace-nowrap">
                              {val.toFixed(4)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </ExpandableSection>

              {/* Research Interpretation Callout */}
              <div className="border border-paper-rule/80 rounded-xl bg-paper-canvas/30 p-4 space-y-1.5 text-xs text-ink-body leading-relaxed">
                <div className="font-semibold text-ink-primary">
                  Microstructure Execution Interpretation
                </div>
                <p className="text-ink-muted">
                  The model estimates a <strong className="text-ink-primary font-medium">{(result.prediction.confidence * 100).toFixed(1)}%</strong> probability of a 5-second {result.prediction.predicted_label} displacement. In live automated market making, executions remain subject to latency queue priority races and adverse selection: resting limit orders at the touch fill at disproportionate rates precisely when incoming market orders sweep liquidity through the book.
                </p>
              </div>
            </div>
          )}
        </section>
      </div>
    </article>
  );
}
