import { ExpandableSection } from "@/components/ExpandableSection";

export default function LimitationsPage() {
  return (
    <article className="space-y-12">
      <header className="rounded-3xl border border-paper-rule bg-paper-canvas/70 p-8 sm:p-10 relative overflow-hidden space-y-5">
        <div className="flex flex-wrap items-center gap-2 text-xs font-sans text-ink-muted">
          <span className="px-3 py-1 rounded-full bg-paper-surface border border-paper-rule text-ink-primary font-semibold text-[11px] uppercase tracking-wide">
            Institutional Limitations Paper
          </span>
          <span className="px-3 py-1 rounded-full bg-paper-surface/60 border border-paper-rule/60 text-[11px]">
            Microstructure Execution Frictions
          </span>
          <span className="px-3 py-1 rounded-full bg-rose-950/20 text-direction-down font-medium border border-rose-950/30 text-[11px]">
            Non-Trading Mandate
          </span>
        </div>

        <div className="space-y-3 max-w-4xl">
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight text-ink-primary leading-tight font-sans">
            Execution Realities, Uncertainty & Structural Limitations
          </h1>
          <p className="text-base sm:text-lg text-ink-body leading-relaxed font-sans font-normal">
            A candid quantitative appraisal of the economic frictions separating offline validation accuracy 
            from live market execution profitability.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3 border-t border-paper-rule/60 text-xs font-sans text-ink-muted">
          <div><span className="text-ink-faint block text-[11px]">Expected Alpha</span><strong className="text-ink-primary font-medium"><span className="numeric-value">5</span> to <span className="numeric-value">15</span> bps (<span className="numeric-value">5s</span> Δmid)</strong></div>
          <div><span className="text-ink-faint block text-[11px]">Exchange Fee Drag</span><strong className="text-ink-primary font-medium"><span className="numeric-value">2</span> to <span className="numeric-value">5</span> bps per Leg</strong></div>
          <div><span className="text-ink-faint block text-[11px]">Operational Status</span><strong className="text-direction-down font-medium">Strictly Unauthorized</strong></div>
        </div>
      </header>

      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-5">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            The Gap Between Statistical Signal and Live Monetization
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Offline predictive accuracy is a prerequisite, not a commercial guarantee.
          </p>
        </div>

        <div className="space-y-4 text-xs leading-relaxed text-ink-body font-sans max-w-3xl">
          <p>
            A common misconception in financial machine learning is assuming that achieving <span className="numeric-value">59.10%</span> classification 
            accuracy across ternary outcomes directly implies an automated trading strategy can generate positive net PnL. 
            In electronic matching engines, statistical alpha is merely a precondition, not a guarantee of profitability.
          </p>
          <div className="rounded-xl border border-brass/25 bg-brass/5 p-4 space-y-1.5">
            <div className="text-xs font-semibold text-brass font-sans">
              Microstructure Expectancy Calculation
            </div>
            <p className="text-xs text-ink-muted leading-relaxed">
              At a <span className="numeric-value">5,000</span>-millisecond forecasting horizon, the expected gross magnitude of mid-price displacement (Δmid) 
              typically ranges between <strong className="text-ink-primary font-semibold"><span className="numeric-value">5</span> and <span className="numeric-value">15</span> basis points</strong>. Four formidable market frictions routinely 
              convert this gross signal into negative net performance.
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-6">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            The Four Critical Market Frictions
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Microstructure barriers preventing naive capitalization of high-frequency forecasts.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
          <div className="rounded-xl border border-paper-rule/70 bg-paper-surface/50 p-6 space-y-3 hover:border-paper-border transition-colors">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-sans px-2.5 py-0.5 rounded-full bg-paper-canvas border border-paper-rule text-ink-faint uppercase font-semibold">
                Friction 01
              </span>
              <span className="text-xs font-sans text-ink-muted">Latency Arbitrage</span>
            </div>
            <h3 className="text-sm font-semibold text-ink-primary font-sans">
              Queue Priority & Latency Arbitrage
            </h3>
            <p className="text-xs text-ink-muted leading-relaxed font-sans">
              PredAlpha operates on discrete 1.0-second sampled order book snapshots. In live cryptocurrency matching 
              engines, however, order book events occur at sub-millisecond frequencies. When the model detects an upward 
              order flow shift, competing high-frequency participants are already joining the queue or canceling resting ask liquidity. 
              Submitting a passive limit order joins the back of the queue; by the time the order reaches priority at the front of the book, 
              the price displacement has frequently already occurred, leaving the strategy with stale executions or missed fills.
            </p>
          </div>

          <div className="rounded-xl border border-paper-rule/70 bg-paper-surface/50 p-6 space-y-3 hover:border-paper-border transition-colors">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-sans px-2.5 py-0.5 rounded-full bg-paper-canvas border border-paper-rule text-ink-faint uppercase font-semibold">
                Friction 02
              </span>
              <span className="text-xs font-sans text-ink-muted">Adverse Selection</span>
            </div>
            <h3 className="text-sm font-semibold text-ink-primary font-sans">
              Adverse Selection & The Maker Touch Decay
            </h3>
            <p className="text-xs text-ink-muted leading-relaxed font-sans">
              Passive maker strategies earn the bid-ask spread but suffer from severe adverse selection. In fast-moving order books, 
              resting limit orders at the touch are executed disproportionately when aggressive institutional flow sweeps the book. 
              In other words, a passive buy order is most likely to be filled precisely when the price is about to collapse through the bid. 
              The strategy experiences the classic &quot;winner&apos;s curse&quot;: fills occur primarily when the prediction is wrong.
            </p>
          </div>

          <div className="rounded-xl border border-paper-rule/70 bg-paper-surface/50 p-6 space-y-3 hover:border-paper-border transition-colors">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-sans px-2.5 py-0.5 rounded-full bg-paper-canvas border border-paper-rule text-ink-faint uppercase font-semibold">
                Friction 03
              </span>
              <span className="text-xs font-sans text-ink-muted">Cost of Liquidity</span>
            </div>
            <h3 className="text-sm font-semibold text-ink-primary font-sans">
              Exchange Transaction Fees & Bid-Ask Drag
            </h3>
            <p className="text-xs text-ink-muted leading-relaxed font-sans">
              To avoid queue waiting, a strategy could cross the spread as an aggressive market taker. However, taker executions pay 
              the full half-spread (10 to 20 basis points) plus explicit exchange taker fees (2 to 5 basis points). Over a 5-second horizon 
              where expected gross alpha is only 5 to 15 bps, the round-trip friction of spread crossing and fees exceeds the entire 
              gross forecast displacement. Unconstrained aggressive trading generates negative net expectancy.
            </p>
          </div>

          <div className="rounded-xl border border-paper-rule/70 bg-paper-surface/50 p-6 space-y-3 hover:border-paper-border transition-colors">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-sans px-2.5 py-0.5 rounded-full bg-paper-canvas border border-paper-rule text-ink-faint uppercase font-semibold">
                Friction 04
              </span>
              <span className="text-xs font-sans text-direction-down font-medium">Distributional Flips</span>
            </div>
            <h3 className="text-sm font-semibold text-ink-primary font-sans">
              The <span className="numeric-value">28.6%</span> Directional Flip Rate
            </h3>
            <p className="text-xs text-ink-muted leading-relaxed font-sans">
              Confusion matrix diagnostics on our <span className="numeric-value">1,428</span>-sample validation cohort establish that <strong className="text-ink-primary"><span className="numeric-value">28.6%</span> of predictions</strong> 
              represent severe directional inversions (predicting an Up move when price drops Down, or vice-versa). In unhedged directional 
              trading, these errors incur maximum losses: entering long at the ask on a market that immediately trends down. 
              Without asymmetric risk gating, these catastrophic flips create heavy negative skewness in backtested returns.
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-6">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            Scientific Demarcation: Proven Findings vs. Open Questions
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Explicit demarcation of verified empirical evidence from unverified assumptions.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="rounded-xl border border-direction-up/25 bg-paper-surface/60 p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-direction-up/20 pb-3">
              <h3 className="text-xs font-sans uppercase tracking-wide text-direction-up font-semibold">
                Empirically Demonstrated Findings
              </h3>
              <span className="text-[10px] font-sans px-2.5 py-0.5 rounded-full bg-direction-up/15 text-direction-up font-medium">
                Verified
              </span>
            </div>
            <ul className="space-y-3.5 text-xs font-sans text-ink-body">
              <li className="flex items-start space-x-2.5">
                <span className="text-direction-up font-sans font-bold leading-none mt-0.5">·</span>
                <span className="leading-relaxed"><strong className="text-ink-primary font-semibold">Causal Signal Extraction:</strong> Sequence modeling over <span className="numeric-value">10</span> seconds of normalized L2 data extracts statistically significant directional signal (<span className="numeric-value">59.10%</span> accuracy, Macro-F1 <span className="numeric-value">0.5543</span>) outperforming random and naive baselines.</span>
              </li>
              <li className="flex items-start space-x-2.5">
                <span className="text-direction-up font-sans font-bold leading-none mt-0.5">·</span>
                <span className="leading-relaxed"><strong className="text-ink-primary font-semibold">Multicollinearity Barriers:</strong> Appending microprice deltas introduces severe collinearity (VIF &gt; <span className="numeric-value">10</span>) that impairs generalization out-of-fold (<span className="numeric-value">-58 bps</span>).</span>
              </li>
              <li className="flex items-start space-x-2.5">
                <span className="text-direction-up font-sans font-bold leading-none mt-0.5">·</span>
                <span className="leading-relaxed"><strong className="text-ink-primary font-semibold">Order Flow Imbalance Redundancy:</strong> On continuous <span className="numeric-value">1s</span> sampled feeds, static touch queue depth provides superior forward predictive value over discrete quote event increments.</span>
              </li>
              <li className="flex items-start space-x-2.5">
                <span className="text-direction-up font-sans font-bold leading-none mt-0.5">·</span>
                <span className="leading-relaxed"><strong className="text-ink-primary font-semibold">Asymmetric Loss Modulation:</strong> Penalizing severe directional flips (λ = <span className="numeric-value">0.50</span>) successfully suppresses inversions by <span className="numeric-value">-9.7</span> events, providing an operational risk trade-off.</span>
              </li>
            </ul>
          </div>

          <div className="rounded-xl border border-brass/25 bg-paper-surface/60 p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-brass/20 pb-3">
              <h3 className="text-xs font-sans uppercase tracking-wide text-brass font-semibold">
                Unproven Hypotheses & Open Questions
              </h3>
              <span className="text-[10px] font-sans px-2.5 py-0.5 rounded-full bg-brass/15 text-brass font-medium">
                Unsealed
              </span>
            </div>
            <ul className="space-y-3.5 text-xs font-sans text-ink-muted">
              <li className="flex items-start space-x-2.5">
                <span className="text-brass font-sans font-bold leading-none mt-0.5">·</span>
                <span className="leading-relaxed"><strong className="text-ink-primary font-semibold">Live Matching Queue Dynamics:</strong> Offline historical simulations cannot model real-time cancellation races and order priority queue decay.</span>
              </li>
              <li className="flex items-start space-x-2.5">
                <span className="text-brass font-sans font-bold leading-none mt-0.5">·</span>
                <span className="leading-relaxed"><strong className="text-ink-primary font-semibold">Net Monetization Survival:</strong> No empirical evidence establishes that positive net PnL can survive real exchange taker fees and adverse selection.</span>
              </li>
              <li className="flex items-start space-x-2.5">
                <span className="text-brass font-sans font-bold leading-none mt-0.5">·</span>
                <span className="leading-relaxed"><strong className="text-ink-primary font-semibold">Regime Invariance:</strong> Model robustness has not been tested across liquidation cascades, funding rate shocks, or macroeconomic announcements.</span>
              </li>
              <li className="flex items-start space-x-2.5">
                <span className="text-brass font-sans font-bold leading-none mt-0.5">·</span>
                <span className="leading-relaxed"><strong className="text-ink-primary font-semibold">Sealed Holdout Performance:</strong> The final holdout dataset (<code className="font-mono text-ink-primary bg-paper-canvas px-1.5 py-0.5 rounded border border-paper-rule">test_scaled.npz</code>) remains sealed to prevent iterative p-hacking.</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-rose-950/30 bg-rose-950/10 p-6 sm:p-8 space-y-3">
        <div className="text-direction-down font-semibold font-sans text-xs uppercase tracking-wide">
          Institutional Governance & Non-Trading Mandate
        </div>
        <p className="text-xs text-ink-body font-sans leading-relaxed">
          PredAlpha is an academic quantitative research project. The models, metrics, and code in this repository 
          are provided solely for scientific inquiry into high-frequency market microstructure. 
          Live trading, capital deployment, and prospective Session 001 execution remain strictly unauthorized.
        </p>
      </section>
    </article>
  );
}
