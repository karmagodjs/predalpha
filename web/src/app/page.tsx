import Link from "next/link";
import { CANONICAL_EXPERIMENT_SUMMARY } from "@/lib/research-data";

export default function HomePage() {
  return (
    <article className="space-y-16">
      {/* 1. Scholarly Editorial Masthead — Open, intentional layout without floating card wrapping */}
      <header className="space-y-6 pb-10 border-b border-paper-rule">
        <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-ink-muted">
          <div className="flex items-center space-x-3">
            <span className="font-semibold text-ink-primary uppercase tracking-wide text-[11px]">
              Empirical Microstructure Research
            </span>
            <span className="text-paper-rule">/</span>
            <span className="text-ink-muted text-[11px]">
              Working Paper · Protocol 2.0
            </span>
          </div>
          <div className="text-[11px] text-ink-faint">
            Multi-Seed Validated (42, 123, 999) · Holdout Sealed
          </div>
        </div>

        <div className="space-y-4 max-w-4xl">
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-ink-primary leading-[1.15]">
            Can order-book microstructure predict short-horizon price movements?
          </h1>

          <p className="text-base sm:text-lg text-ink-body leading-relaxed max-w-3xl font-normal">
            An empirical investigation of 5-second mid-price directional forecasting using causal recurrent 
            architectures and continuous Level-2 order book event streams under chronological purged walk-forward validation.
          </p>
        </div>

        {/* Technical Metadata Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 pt-6 border-t border-paper-rule/70 text-xs">
          <div>
            <span className="text-ink-faint block text-[11px] uppercase tracking-wide mb-1">Target Instrument</span>
            <span className="text-ink-primary font-medium">BTC/USDT Perpetual</span>
          </div>
          <div>
            <span className="text-ink-faint block text-[11px] uppercase tracking-wide mb-1">Forecast Horizon</span>
            <span className="text-ink-primary font-medium numeric-value">5,000 ms (Δmid)</span>
          </div>
          <div>
            <span className="text-ink-faint block text-[11px] uppercase tracking-wide mb-1">Validation Cohort</span>
            <span className="text-ink-primary font-medium numeric-value">1,428 Continuous Sequences</span>
          </div>
          <div>
            <span className="text-ink-faint block text-[11px] uppercase tracking-wide mb-1">Holdout State</span>
            <span className="text-brass font-medium">Sealed on Disk (test_scaled.npz)</span>
          </div>
        </div>
      </header>

      {/* 2. Executive Research Briefing — Integrated telemetry panel rather than repetitive cards */}
      <section aria-labelledby="benchmark-summary" className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2">
          <div>
            <h2 id="benchmark-summary" className="text-xs uppercase tracking-wide font-semibold text-ink-muted">
              Primary Benchmark Findings · Multi-Seed Out-of-Sample Evaluation
            </h2>
            <p className="text-xs text-ink-faint mt-0.5">
              Averaged across seeds 42, 123, and 999 on 1,428 chronological out-of-fold validation sequences.
            </p>
          </div>
          <div className="text-[11px] text-ink-faint">
            Chronological Walk-Forward · 10s Purged Warmup
          </div>
        </div>

        {/* Integrated Data Panel with Subtle Internal Grid Dividers */}
        <div className="border border-paper-rule bg-paper-canvas/60 rounded-xl overflow-hidden divide-y divide-paper-rule sm:divide-y-0 sm:grid sm:grid-cols-2 lg:grid-cols-4 sm:divide-x sm:divide-paper-rule">
          {/* Stat 1: Primary Accuracy */}
          <div className="p-6 space-y-2">
            <div className="flex items-baseline justify-between">
              <span className="text-xs uppercase tracking-wide font-medium text-ink-muted">Ternary Accuracy</span>
              <span className="text-[11px] text-direction-up font-medium numeric-value">+15.96% vs Majority</span>
            </div>
            <div className="text-3xl sm:text-4xl font-bold tracking-tight text-ink-primary numeric-value">
              59.10%
            </div>
            <p className="text-xs text-ink-muted leading-relaxed">
              Exceeds 43.14% majority-class and 33.33% random baseline with zero forward lookahead.
            </p>
          </div>

          {/* Stat 2: Macro-F1 */}
          <div className="p-6 space-y-2">
            <div className="flex items-baseline justify-between">
              <span className="text-xs uppercase tracking-wide font-medium text-ink-muted">Macro-F1 Score</span>
              <span className="text-[11px] text-ink-faint">Ternary Balance</span>
            </div>
            <div className="text-3xl sm:text-4xl font-bold tracking-tight text-ink-primary numeric-value">
              0.5543
            </div>
            <p className="text-xs text-ink-muted leading-relaxed">
              Harmonic mean balanced across Down (0.643), Stationary (0.354), and Up (0.666).
            </p>
          </div>

          {/* Stat 3: Directional Inversions */}
          <div className="p-6 space-y-2">
            <div className="flex items-baseline justify-between">
              <span className="text-xs uppercase tracking-wide font-medium text-ink-muted">Inversion Flips</span>
              <span className="text-[11px] text-direction-up font-medium numeric-value">-9.7 Flips (λ = 0.50)</span>
            </div>
            <div className="text-3xl sm:text-4xl font-bold tracking-tight text-ink-primary numeric-value">
              399.0
            </div>
            <p className="text-xs text-ink-muted leading-relaxed">
              Severe sign errors reduced from 408.7 to 399.0 under asymmetric penalty loss.
            </p>
          </div>

          {/* Stat 4: Network Capacity */}
          <div className="p-6 space-y-2">
            <div className="flex items-baseline justify-between">
              <span className="text-xs uppercase tracking-wide font-medium text-ink-muted">Capacity</span>
              <span className="text-[11px] text-ink-faint">1-Layer LSTM</span>
            </div>
            <div className="text-3xl sm:text-4xl font-bold tracking-tight text-ink-primary numeric-value">
              5,859
            </div>
            <p className="text-xs text-ink-muted leading-relaxed">
              Parsimonious topology acting as a structural inductive bias against high-frequency noise.
            </p>
          </div>
        </div>
      </section>

      {/* 3. Model Architecture & Pipeline Topology — Dedicated full-width diagram with exact mathematical derivation */}
      <section aria-labelledby="architecture-heading" className="space-y-6 pt-2">
        <div className="space-y-1">
          <h2 id="architecture-heading" className="text-lg sm:text-xl font-bold tracking-tight text-ink-primary">
            Causal Recurrent Pipeline & Tensor Accounting
          </h2>
          <p className="text-xs text-ink-muted">
            Continuous sequence unrolling from 1.0s Level-2 snapshots to 5-second ternary prediction vector.
          </p>
        </div>

        {/* Technical Architecture Schematic */}
        <div className="border border-paper-rule bg-paper-canvas/40 rounded-xl p-5 sm:p-6 overflow-x-auto">
          <svg
            viewBox="0 0 760 178"
            className="w-full min-w-[680px] text-ink-primary select-none font-sans text-xs"
            aria-label="Causal Recurrent Neural Network Pipeline Schematic"
          >
            <defs>
              <marker
                id="arch-arrowhead"
                markerWidth="6"
                markerHeight="6"
                refX="5"
                refY="3"
                orient="auto"
              >
                <polygon points="0 1, 5 3, 0 5" fill="#d99b26" />
              </marker>
            </defs>

            {/* Stage 1: Input Sequence */}
            <g transform="translate(14, 10)">
              <rect
                x="0"
                y="0"
                width="160"
                height="156"
                rx="6"
                fill="#151921"
                stroke="#2b3240"
                strokeWidth="1"
              />
              <text x="80" y="27" textAnchor="middle" fill="#f3f5f8" fontWeight="600" fontSize="11" fontFamily="var(--font-montserrat), Montserrat, sans-serif">
                Causal Input
              </text>
              <text x="80" y="42" textAnchor="middle" fill="#d99b26" fontSize="10" fontFamily="var(--font-mono), 'JetBrains Mono', monospace">
                Tensor [B, 10, 11]
              </text>
              <line x1="14" y1="51" x2="146" y2="51" stroke="#242934" strokeWidth="1" />
              <text x="18" y="71" fill="#8890a0" fontSize="10" fontFamily="var(--font-inter), -apple-system, sans-serif">L = 10 steps (10s)</text>
              <text x="18" y="89" fill="#8890a0" fontSize="10" fontFamily="var(--font-inter), -apple-system, sans-serif">D = 11 L2 features</text>
              <text x="18" y="107" fill="#8890a0" fontSize="10" fontFamily="var(--font-montserrat), Montserrat, sans-serif">StandardScaled</text>
              <text x="18" y="125" fill="#8890a0" fontSize="10" fontFamily="var(--font-montserrat), Montserrat, sans-serif">Zero lookahead</text>
              <text x="18" y="145" fill="#575e6d" fontSize="9" fontFamily="var(--font-mono), 'JetBrains Mono', monospace">Snapshots t-9 ... t0</text>
            </g>

            {/* Connector 1 */}
            <line x1="174" y1="88" x2="202" y2="88" stroke="#d99b26" strokeWidth="1.5" markerEnd="url(#arch-arrowhead)" />

            {/* Stage 2: Recurrent Core */}
            <g transform="translate(204, 10)">
              <rect
                x="0"
                y="0"
                width="160"
                height="156"
                rx="6"
                fill="#151921"
                stroke="#2b3240"
                strokeWidth="1"
              />
              <text x="80" y="27" textAnchor="middle" fill="#f3f5f8" fontWeight="600" fontSize="11" fontFamily="var(--font-montserrat), Montserrat, sans-serif">
                SmallLSTM Cell
              </text>
              <text x="80" y="42" textAnchor="middle" fill="#d99b26" fontSize="10" fontFamily="var(--font-mono), 'JetBrains Mono', monospace">
                1 Layer · H = 32
              </text>
              <line x1="14" y1="51" x2="146" y2="51" stroke="#242934" strokeWidth="1" />
              <text x="18" y="71" fill="#8890a0" fontSize="10" fontFamily="var(--font-mono), 'JetBrains Mono', monospace">W_ih: [128, 11]</text>
              <text x="18" y="89" fill="#8890a0" fontSize="10" fontFamily="var(--font-mono), 'JetBrains Mono', monospace">W_hh: [128, 32]</text>
              <text x="18" y="107" fill="#8890a0" fontSize="10" fontFamily="var(--font-mono), 'JetBrains Mono', monospace">biases: [256]</text>
              <text x="18" y="125" fill="#8890a0" fontSize="10" fontFamily="var(--font-inter), -apple-system, sans-serif">Params: 5,760</text>
              <text x="18" y="145" fill="#575e6d" fontSize="9" fontFamily="var(--font-inter), -apple-system, sans-serif">Terminal state h_10</text>
            </g>

            {/* Connector 2 */}
            <line x1="364" y1="88" x2="392" y2="88" stroke="#d99b26" strokeWidth="1.5" markerEnd="url(#arch-arrowhead)" />

            {/* Stage 3: Regularization & Projection */}
            <g transform="translate(394, 10)">
              <rect
                x="0"
                y="0"
                width="160"
                height="156"
                rx="6"
                fill="#151921"
                stroke="#2b3240"
                strokeWidth="1"
              />
              <text x="80" y="27" textAnchor="middle" fill="#f3f5f8" fontWeight="600" fontSize="11" fontFamily="var(--font-montserrat), Montserrat, sans-serif">
                Linear Projection
              </text>
              <text x="80" y="42" textAnchor="middle" fill="#d99b26" fontSize="10" fontFamily="var(--font-mono), 'JetBrains Mono', monospace">
                Linear [32 → 3]
              </text>
              <line x1="14" y1="51" x2="146" y2="51" stroke="#242934" strokeWidth="1" />
              <text x="18" y="71" fill="#8890a0" fontSize="10" fontFamily="var(--font-inter), -apple-system, sans-serif">Dropout(p=0.10)</text>
              <text x="18" y="89" fill="#8890a0" fontSize="10" fontFamily="var(--font-mono), 'JetBrains Mono', monospace">W_fc: [3, 32]</text>
              <text x="18" y="107" fill="#8890a0" fontSize="10" fontFamily="var(--font-mono), 'JetBrains Mono', monospace">b_fc: [3]</text>
              <text x="18" y="125" fill="#8890a0" fontSize="10" fontFamily="var(--font-inter), -apple-system, sans-serif">Params: 99</text>
              <text x="18" y="145" fill="#575e6d" fontSize="9" fontFamily="var(--font-mono), 'JetBrains Mono', monospace">Logits: z ∈ ℝ³</text>
            </g>

            {/* Connector 3 */}
            <line x1="554" y1="88" x2="582" y2="88" stroke="#d99b26" strokeWidth="1.5" markerEnd="url(#arch-arrowhead)" />

            {/* Stage 4: Softmax Classification */}
            <g transform="translate(584, 10)">
              <rect
                x="0"
                y="0"
                width="160"
                height="156"
                rx="6"
                fill="#151921"
                stroke="#2b3240"
                strokeWidth="1"
              />
              <text x="80" y="27" textAnchor="middle" fill="#f3f5f8" fontWeight="600" fontSize="11" fontFamily="var(--font-montserrat), Montserrat, sans-serif">
                Softmax Output
              </text>
              <text x="80" y="42" textAnchor="middle" fill="#d99b26" fontSize="10" fontFamily="var(--font-mono), 'JetBrains Mono', monospace">
                P(Δmid)
              </text>
              <line x1="14" y1="51" x2="146" y2="51" stroke="#242934" strokeWidth="1" />
              <text x="18" y="71" fill="#f43f5e" fontWeight="600" fontSize="10" fontFamily="var(--font-inter), -apple-system, sans-serif">Down (Class 0)</text>
              <text x="18" y="89" fill="#8890a0" fontWeight="600" fontSize="10" fontFamily="var(--font-inter), -apple-system, sans-serif">Flat (Class 1)</text>
              <text x="18" y="107" fill="#10b981" fontWeight="600" fontSize="10" fontFamily="var(--font-inter), -apple-system, sans-serif">Up (Class 2)</text>
              <text x="18" y="125" fill="#8890a0" fontSize="10" fontFamily="var(--font-montserrat), Montserrat, sans-serif">Top-1 Decision</text>
              <text x="18" y="145" fill="#575e6d" fontSize="9" fontFamily="var(--font-mono), 'JetBrains Mono', monospace">Gating: τ ≥ 0.65</text>
            </g>
          </svg>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs text-ink-muted gap-2 px-1">
          <p className="max-w-3xl leading-relaxed">
            <strong className="text-ink-primary font-medium">Exact Parameter Derivation:</strong> 4 × (11 × 32 + 32² + 32) = 5,760 recurrent weights, plus 32 × 3 + 3 = 99 projection weights equals exactly <span className="numeric-value font-semibold text-ink-primary">5,859</span> parameters.
          </p>
          <div className="font-mono text-[11px] text-ink-faint shrink-0">
            Memory: ~2.8 ms CPU latency
          </div>
        </div>
      </section>

      {/* 4. Comparative Scientific Findings — Editorial narrative balancing validated results vs rejected hypotheses */}
      <section aria-labelledby="findings-heading" className="space-y-6 pt-4">
        <div className="space-y-1">
          <h2 id="findings-heading" className="text-lg sm:text-xl font-bold tracking-tight text-ink-primary">
            Empirical Findings & Structural Trade-offs
          </h2>
          <p className="text-xs text-ink-muted">
            Demarcation of verified predictive mechanisms from empirically rejected hypotheses.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Column A: What Works */}
          <div className="space-y-6 border-t border-direction-up/30 pt-4">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-direction-up uppercase tracking-wide text-[11px]">
                Empirically Demonstrated Findings
              </span>
              <span className="text-direction-up text-[11px] font-mono">Confirmed</span>
            </div>

            <div className="space-y-4 text-xs text-ink-body leading-relaxed">
              <div className="space-y-1.5">
                <div className="font-semibold text-ink-primary">
                  1. Causal Sequential Memory vs Instantaneous Snapshot
                </div>
                <p className="text-ink-muted">
                  Consuming 10 rolling one-second snapshots unrolled chronologically extracts genuine quotation drift, yielding 59.10% ternary accuracy (+15.96% over majority class 43.14%) without lookahead bias.
                </p>
              </div>

              <div className="space-y-1.5">
                <div className="font-semibold text-ink-primary">
                  2. Asymmetric Loss Penalty Suppresses Catastrophic Flips
                </div>
                <p className="text-ink-muted">
                  Penalizing extreme sign errors (predicting Up when price collapses Down) with λ = 0.50 reduced mean directional inversions by -9.7 events (from 408.7 to 399.0), establishing an operational risk mitigation trade-off.
                </p>
              </div>

              <div className="space-y-1.5">
                <div className="font-semibold text-ink-primary">
                  3. Confidence Threshold Gating for Execution
                </div>
                <p className="text-ink-muted">
                  Restricting execution to predictions with softmax confidence τ ≥ 0.65 increases precision to 64.5% (23.9% coverage), while τ ≥ 0.80 achieves 85.7% accuracy at 1.0% market participation.
                </p>
              </div>
            </div>
          </div>

          {/* Column B: What Fails / Structural Caveats */}
          <div className="space-y-6 border-t border-paper-rule pt-4">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-brass uppercase tracking-wide text-[11px]">
                Rejected Hypotheses & Market Realities
              </span>
              <span className="text-brass text-[11px] font-mono">Constrained</span>
            </div>

            <div className="space-y-4 text-xs text-ink-body leading-relaxed">
              <div className="space-y-1.5">
                <div className="font-semibold text-ink-primary">
                  1. The Microprice Collinearity Barrier
                </div>
                <p className="text-ink-muted">
                  Appending microprice deviation (Δmicro) pushed Variance Inflation Factors (VIF) past 10.0, destabilizing weights and reducing out-of-fold accuracy by -58 bps (58.52% vs 59.10%).
                </p>
              </div>

              <div className="space-y-1.5">
                <div className="font-semibold text-ink-primary">
                  2. Quote OFI Redundancy at 1-Second Sampling
                </div>
                <p className="text-ink-muted">
                  Adding quote-level order flow increments degraded accuracy by -86 bps. Replacing depth imbalance with OFI collapsed accuracy by -278 bps. Instantaneous touch depth provides superior forward state information over discrete quote counts.
                </p>
              </div>

              <div className="space-y-1.5">
                <div className="font-semibold text-ink-primary">
                  3. Unhedged Flips & Execution Fee Drag
                </div>
                <p className="text-ink-muted">
                  28.6% of predictions remain directional inversions. In live markets, taker fee drag (2–5 bps) and adverse selection at the touch routinely convert gross 5–15 bps forecast displacements into negative net PnL.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. Canonical 6-Arm Experiment Results Table */}
      <section aria-labelledby="experiment-table-heading" className="space-y-4 pt-4">
        <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2">
          <div>
            <h2 id="experiment-table-heading" className="text-lg sm:text-xl font-bold tracking-tight text-ink-primary">
              Canonical 6-Arm Evaluation Summary
            </h2>
            <p className="text-xs text-ink-muted">
              Out-of-sample averages across 6 controlled evaluation arms (1,428 continuous sequences per seed).
            </p>
          </div>
          <Link
            href="/experiments"
            className="text-xs text-brass hover:underline font-medium"
          >
            Full multi-seed breakdown & diagnostics →
          </Link>
        </div>

        <div className="border border-paper-rule rounded-xl overflow-hidden bg-paper-canvas/30">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="bg-paper-surface/60 border-b border-paper-rule text-ink-muted text-[11px]">
                  <th className="py-3 px-4 font-semibold">Evaluation Arm</th>
                  <th className="py-3 px-3 text-center">Inputs (D)</th>
                  <th className="py-3 px-3 text-right">OOS Accuracy</th>
                  <th className="py-3 px-3 text-right">Macro-F1</th>
                  <th className="py-3 px-3 text-right">Inversions</th>
                  <th className="py-3 px-3 text-right">Δ Acc (bps)</th>
                  <th className="py-3 px-4 text-left font-semibold">Empirical Finding</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-paper-rule/60 text-ink-body">
                {CANONICAL_EXPERIMENT_SUMMARY.map((row, idx) => (
                  <tr
                    key={idx}
                    className={`hover:bg-paper-surface/40 transition-colors ${
                      row.status === "BASELINE"
                        ? "bg-paper-surface/20 font-medium"
                        : row.status === "TRADE-OFF"
                        ? "bg-brass/5"
                        : ""
                    }`}
                  >
                    <td className="py-3 px-4 font-semibold text-ink-primary whitespace-nowrap">
                      {row.arm}
                    </td>
                    <td className="py-3 px-3 text-center text-ink-muted">{row.features}</td>
                    <td className="py-3 px-3 text-right font-bold text-ink-primary tabular-nums">
                      {row.accuracy.toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums text-ink-primary">
                      {row.macroF1.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums text-ink-muted">
                      {row.inversions.toFixed(1)}
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums">
                      <span
                        className={`font-semibold ${
                          row.deltaBps > 0
                            ? "text-direction-up"
                            : row.deltaBps < 0
                            ? "text-direction-down"
                            : "text-ink-faint"
                        }`}
                      >
                        {row.deltaBps > 0 ? `+${row.deltaBps}` : row.deltaBps}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-[11px] text-ink-muted">
                      {row.finding}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* 6. Analytical Model Lab Gateway — Restrained invitation without generic marketing boxes */}
      <section className="pt-6 pb-4 border-t border-paper-rule">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2 max-w-xl">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-brass">
              Interactive Research Tool
            </div>
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary">
              Evaluate PyTorch Inference in the Model Lab
            </h2>
            <p className="text-xs text-ink-muted leading-relaxed">
              Test model parameter sensitivity across 4 pre-configured market regimes (Bid Wall, Ask Overhang, Balanced Spread, Volatility Shock) or modulate terminal order book features with direct tensor inspection.
            </p>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <Link
              href="/lab"
              className="px-5 py-2.5 text-xs font-semibold text-paper-bg bg-ink-primary hover:bg-ink-body rounded-lg transition-colors"
            >
              Launch Model Lab →
            </Link>
            <Link
              href="/methodology"
              className="px-4 py-2.5 text-xs font-medium text-ink-muted hover:text-ink-primary rounded-lg border border-paper-rule hover:border-paper-border hover:bg-paper-surface/50 transition-colors"
            >
              Replication Guide
            </Link>
          </div>
        </div>
      </section>
    </article>
  );
}
