"use client";

import { useState } from "react";
import { ConfusionMatrix } from "@/components/ConfusionMatrix";
import { CoverageCurveChart } from "@/components/CoverageCurveChart";
import { ExpandableSection } from "@/components/ExpandableSection";
import {
  CONFUSION_MATRICES,
  VALIDATION_COHORT_META,
} from "@/lib/research-data";

export default function ValidationPage() {
  const [selectedCmIndex, setSelectedCmIndex] = useState(0);
  const activeCm = CONFUSION_MATRICES[selectedCmIndex];

  return (
    <article className="space-y-12">
      <header className="rounded-3xl border border-paper-rule bg-paper-canvas/70 p-8 sm:p-10 relative overflow-hidden space-y-5">
        <div className="flex flex-wrap items-center gap-2 text-xs font-sans text-ink-muted">
          <span className="px-3 py-1 rounded-full bg-paper-surface border border-paper-rule text-ink-primary font-semibold text-[11px] uppercase tracking-wide">
            Empirical Validation Paper
          </span>
          <span className="px-3 py-1 rounded-full bg-paper-surface/60 border border-paper-rule/60 text-[11px]">
            Diagnostic Error Analysis
          </span>
          <span className="px-3 py-1 rounded-full bg-brass/10 border border-brass/25 text-brass font-medium text-[11px]">
            Cohort N = 1,428
          </span>
        </div>

        <div className="space-y-3 max-w-4xl">
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight text-ink-primary leading-tight font-sans">
            Validation Diagnostics & Empirical Error Matrix
          </h1>
          <p className="text-base sm:text-lg text-ink-body leading-relaxed font-sans font-normal">
            Detailed error distribution, directional inversion flip risks, and selective classification 
            confidence gating evaluated on 1,428 sequentially continuous out-of-fold validation sequences.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3 border-t border-paper-rule/60 text-xs font-sans text-ink-muted">
          <div><span className="text-ink-faint block text-[11px]">Warmup Purging</span><strong className="text-ink-primary font-medium">10 Steps (10s Excluded)</strong></div>
          <div><span className="text-ink-faint block text-[11px]">Split Architecture</span><strong className="text-ink-primary font-medium">Chronological Walk-Forward</strong></div>
          <div><span className="text-ink-faint block text-[11px]">Leakage Control</span><strong className="text-direction-up font-medium">Zero Forward Overlap</strong></div>
        </div>
      </header>

      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-6">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            Validation Cohort Class Balance
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Continuous Level-2 sequence window spanning 1,428 consecutive seconds.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-6">
          <div className="rounded-xl border border-paper-rule/70 bg-paper-surface/50 p-5 space-y-3">
            <div className="flex items-center justify-between text-[11px] font-sans text-ink-muted">
              <span>Class 0 · Down (&lt; -θ)</span>
              <span className="numeric-value text-direction-down font-medium">43.14%</span>
            </div>
            <div className="text-2xl sm:text-3xl font-bold numeric-value text-direction-down">
              {VALIDATION_COHORT_META.downCount}
            </div>
            <div className="w-full bg-paper-rule h-1.5 rounded-full overflow-hidden">
              <div className="bg-direction-down h-full rounded-full transition-[width] duration-300 ease-out" style={{ width: "43.14%" }} />
            </div>
            <p className="text-xs text-ink-muted font-sans leading-relaxed">
              Negative 5s price displacement exceeding threshold θ.
            </p>
          </div>

          <div className="rounded-xl border border-paper-rule/70 bg-paper-surface/50 p-5 space-y-3">
            <div className="flex items-center justify-between text-[11px] font-sans text-ink-muted">
              <span>Class 1 · Stationary (|Δ| ≤ θ)</span>
              <span className="numeric-value text-ink-body font-medium">13.66%</span>
            </div>
            <div className="text-2xl sm:text-3xl font-bold numeric-value text-ink-primary">
              {VALIDATION_COHORT_META.flatCount}
            </div>
            <div className="w-full bg-paper-rule h-1.5 rounded-full overflow-hidden">
              <div className="bg-ink-muted h-full rounded-full transition-[width] duration-300 ease-out" style={{ width: "13.66%" }} />
            </div>
            <p className="text-xs text-ink-muted font-sans leading-relaxed">
              Sub-threshold quotation micro-drift or balance.
            </p>
          </div>

          <div className="rounded-xl border border-paper-rule/70 bg-paper-surface/50 p-5 space-y-3">
            <div className="flex items-center justify-between text-[11px] font-sans text-ink-muted">
              <span>Class 2 · Up (&gt; θ)</span>
              <span className="numeric-value text-direction-up font-medium">43.21%</span>
            </div>
            <div className="text-2xl sm:text-3xl font-bold numeric-value text-direction-up">
              {VALIDATION_COHORT_META.upCount}
            </div>
            <div className="w-full bg-paper-rule h-1.5 rounded-full overflow-hidden">
              <div className="bg-direction-up h-full rounded-full transition-[width] duration-300 ease-out" style={{ width: "43.21%" }} />
            </div>
            <p className="text-xs text-ink-muted font-sans leading-relaxed">
              Positive 5s price displacement exceeding threshold θ.
            </p>
          </div>
        </div>

        <p className="text-xs text-ink-muted leading-relaxed font-sans">
          The natural market regime exhibits marked directional dominance over stationarity at 5 seconds: continuous quotation activity in high-frequency matching engines generates continuous sub-tick displacement, making pure stationary regimes relatively infrequent.
        </p>
      </section>

      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-6">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            Empirical Confusion Matrix & Inversion Asymmetry
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Out-of-fold predictions vs actual realized 5s displacements across evaluation arms.
          </p>
        </div>

        <div className="flex flex-wrap gap-2 text-xs font-sans">
          {CONFUSION_MATRICES.map((cm, idx) => (
            <button
              key={cm.name}
              onClick={() => setSelectedCmIndex(idx)}
              className={`px-4 py-2 rounded-full border transition-colors text-xs font-medium font-sans ${
                selectedCmIndex === idx
                  ? "bg-paper-surface text-ink-primary font-semibold border-paper-border"
                  : "text-ink-muted border-paper-rule hover:text-ink-primary hover:bg-paper-surface/50"
              }`}
            >
              {cm.name}
            </button>
          ))}
        </div>

        <ConfusionMatrix matrix={activeCm.matrix} title={activeCm.name} />

        <div className="rounded-xl border border-rose-950/40 bg-rose-950/15 p-5 space-y-3 text-xs leading-relaxed text-ink-body font-sans">
          <div className="flex items-center justify-between text-direction-down font-semibold">
            <span>Severe Directional Flip Rate: <span className="numeric-value">28.6%</span></span>
            <span className="text-[11px] px-2 py-0.5 rounded-full bg-rose-950/30 border border-direction-down/20 font-sans">
              Execution Risk
            </span>
          </div>
          <p className="text-xs text-ink-muted">
            While ternary accuracy reaches <span className="numeric-value">59.10%</span>, inspection of extreme off-diagonal cells reveals that 
            <strong className="text-ink-primary"> <span className="numeric-value">28.6%</span> of predictions</strong> represent severe directional inversions:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs bg-paper-canvas/60 p-3 rounded-lg border border-paper-rule/60">
            <div>Actual Down predicted as Up: <strong className="text-direction-down font-bold numeric-value">215</strong> (<span className="numeric-value">34.9%</span> error rate)</div>
            <div>Actual Up predicted as Down: <strong className="text-direction-down font-bold numeric-value">193</strong> (<span className="numeric-value">31.3%</span> error rate)</div>
          </div>
          <p className="text-xs text-ink-muted leading-relaxed">
            In automated trade execution, directional flips incur double transaction drag: crossing the spread to enter long on an aggressively collapsing market generates maximum adverse slippage. This error concentration demonstrates why raw model signals cannot be traded without confidence thresholds or asymmetric execution rules.
          </p>
        </div>
      </section>

      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-6">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            Selective Classification: Confidence Threshold Gating
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Trading turnover vs. prediction accuracy trade-off under execution cutoffs τ ∈ [<span className="numeric-value">0.40</span>, <span className="numeric-value">0.85</span>].
          </p>
        </div>

        <p className="text-xs text-ink-body font-sans leading-relaxed">
          An algorithmic execution strategy is under no obligation to trade every second. By applying a confidence threshold τ, 
          low-conviction predictions are filtered out, achieving monotonic accuracy gains at the expense of market participation:
        </p>

        <CoverageCurveChart />

        <ExpandableSection
          title="Analysis: The Precision vs. Turnover Trade-Off"
          summary="Why 85.7% selective accuracy does not guarantee commercial liquidity monetization"
          badge="Market Participation Mechanics"
        >
          <div className="space-y-2 text-xs leading-relaxed text-ink-body font-sans">
            <p>
              When unconstrained (τ ≥ <span className="numeric-value">0.40</span>), the model participates in <strong className="text-ink-primary font-semibold numeric-value">97.5%</strong> of market seconds at 
              <strong className="text-ink-primary font-semibold"> <span className="numeric-value">59.6%</span> accuracy</strong>. As the confidence filter is tightened:
            </p>
            <ul className="space-y-1 text-xs text-ink-muted list-disc list-inside bg-paper-surface/40 p-3 rounded-lg border border-paper-rule/60">
              <li>τ ≥ <span className="numeric-value">0.60</span>: Accuracy climbs to <span className="numeric-value">64.5%</span> (<span className="numeric-value">23.9%</span> decision coverage; ~1 trade per 4 seconds)</li>
              <li>τ ≥ <span className="numeric-value">0.70</span>: Accuracy reaches <span className="numeric-value">72.7%</span> (<span className="numeric-value">5.4%</span> decision coverage; ~1 trade per 18 seconds)</li>
              <li>τ ≥ <span className="numeric-value">0.80</span>: Accuracy achieves <span className="numeric-value">85.7%</span> (<span className="numeric-value">1.0%</span> decision coverage; <span className="numeric-value">14</span> total trades across <span className="numeric-value">1,428</span> seconds)</li>
            </ul>
            <p className="text-ink-muted">
              While 85.7% selective accuracy appears exceptional, the extreme drop in turnover (1 trade every 100 seconds) 
              means capital utilization is severely restricted. Practical quantitative strategies must balance signal precision 
              against the opportunity cost of idle trading capital.
            </p>
          </div>
        </ExpandableSection>
      </section>
    </article>
  );
}
