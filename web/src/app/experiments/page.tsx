import {
  CANONICAL_EXPERIMENT_SUMMARY,
  EXP1_SEEDS,
  EXP2_SEEDS,
  EXP3_SEEDS,
} from "@/lib/research-data";
import { ExpandableSection } from "@/components/ExpandableSection";

export default function ExperimentsPage() {
  return (
    <article className="space-y-12">
      {/* 1. Header in Rounded-3xl Panel */}
      <header className="rounded-3xl border border-paper-rule bg-paper-canvas/70 p-8 sm:p-10 space-y-5">
        <div className="flex flex-wrap items-center gap-2 text-xs font-sans text-ink-muted">
          <span className="px-3 py-1 rounded-full bg-paper-surface border border-paper-rule text-ink-primary font-semibold text-[11px] uppercase tracking-wide">
            Ablation Research
          </span>
          <span className="px-3 py-1 rounded-full bg-paper-surface/60 border border-paper-rule/60 text-[11px]">
            Multi-Seed Protocol
          </span>
          <span className="px-3 py-1 rounded-full bg-brass/10 border border-brass/25 text-brass font-medium text-[11px]">
            Seeds 42, 123, 999
          </span>
        </div>

        <div className="space-y-3 max-w-4xl">
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight text-ink-primary leading-tight font-sans">
            Controlled Ablation Studies & Empirical Hypotheses
          </h1>
          <p className="text-base sm:text-lg text-ink-body leading-relaxed font-sans font-normal">
            Three pre-declared controlled experiments evaluated across identical chronological data partitions 
            to isolate the causal impact of loss formulations, feature space expansions, and order flow metrics.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3 text-xs text-ink-muted font-sans border-t border-paper-rule/60">
          <div><span className="text-ink-faint block text-[11px]">Cohort</span><strong className="text-ink-primary font-medium">1,428 Sequences per Fold</strong></div>
          <div><span className="text-ink-faint block text-[11px]">Evaluation Mode</span><strong className="text-ink-primary font-medium">Out-of-Sample Holdout</strong></div>
          <div><span className="text-ink-faint block text-[11px]">Protocol</span><strong className="text-brass font-medium">Pre-declared Frozen Phase 27</strong></div>
        </div>
      </header>

      {/* 2. Canonical 6-Arm Summary Table in Rounded-2xl Frame with Visual Delta Bars */}
      <section className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            Canonical 6-Arm Evaluation Results
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Multi-seed averages computed across random seeds 42, 123, and 999 on 1,428 chronological out-of-fold validation sequences.
          </p>
        </div>

        <div className="rounded-2xl border border-paper-rule overflow-hidden bg-paper-canvas/40">
          <div className="overflow-x-auto">
            <table className="w-full text-left font-sans text-xs">
              <thead>
                <tr className="bg-paper-surface/80 border-b border-paper-rule text-ink-muted text-[11px]">
                  <th className="py-3 px-4 font-sans font-semibold">Model Arm</th>
                  <th className="py-3 px-3 text-center">D</th>
                  <th className="py-3 px-3 text-right">OOS Accuracy</th>
                  <th className="py-3 px-3 text-right">Macro-F1</th>
                  <th className="py-3 px-3 text-right">Inversions</th>
                  <th className="py-3 px-3 text-right">Δ Acc (bps)</th>
                  <th className="py-3 px-4 text-left font-sans font-semibold">Empirical Verdict</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-paper-rule/60 text-ink-body">
                {CANONICAL_EXPERIMENT_SUMMARY.map((row, idx) => (
                  <tr
                    key={idx}
                    className={`hover:bg-paper-surface/40 transition-colors ${
                      row.status === "BASELINE"
                        ? "bg-paper-surface/30 font-semibold"
                        : row.status === "TRADE-OFF"
                        ? "bg-brass/5"
                        : ""
                    }`}
                  >
                    <td className="py-3 px-4 font-semibold text-ink-primary whitespace-nowrap font-sans">
                      {row.arm}
                    </td>
                    <td className="py-3 px-3 text-center text-ink-muted numeric-value">{row.features}</td>
                    <td className="py-3 px-3 text-right font-bold text-ink-primary numeric-value">
                      {row.accuracy.toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right numeric-value text-ink-primary">
                      {row.macroF1.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 text-right numeric-value text-ink-muted">
                      {row.inversions.toFixed(1)}
                    </td>
                    <td className="py-3 px-3 text-right numeric-value">
                      <div className="flex items-center justify-end space-x-1.5">
                        <span
                          className={`font-semibold numeric-value ${
                            row.deltaBps > 0
                              ? "text-direction-up"
                              : row.deltaBps < 0
                              ? "text-direction-down"
                              : "text-ink-faint"
                          }`}
                        >
                          {row.deltaBps > 0 ? `+${row.deltaBps}` : row.deltaBps}
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-xs font-sans text-ink-muted">
                      {row.status === "BASELINE" && (
                        <span className="text-ink-primary font-medium px-2.5 py-0.5 rounded-full bg-paper-surface border border-paper-rule">
                          Validated Benchmark
                        </span>
                      )}
                      {row.status === "TRADE-OFF" && (
                        <span className="text-brass font-medium px-2.5 py-0.5 rounded-full bg-brass/10 border border-brass/30">
                          Safety Trade-Off (-9.7 Flips)
                        </span>
                      )}
                      {row.status === "REJECTED" && (
                        <span className="text-direction-down px-2.5 py-0.5 rounded-full bg-direction-down/10 border border-direction-down/20">
                          Empirically Rejected
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <p className="text-[11px] text-ink-faint font-sans leading-normal px-2">
          * D = Input feature dimensions. Inversions = Mean severe directional flips (Down predicted as Up, or Up predicted as Down). 
          Δ Acc = Basis points displacement relative to the 11-feature baseline benchmark.
        </p>
      </section>

      {/* 3. Study 1: Loss Formulations */}
      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-6">
        <div className="space-y-1">
          <div className="flex items-center space-x-2 text-xs font-sans uppercase tracking-wide text-brass font-semibold">
            <span className="px-2.5 py-0.5 rounded-full bg-brass/10 border border-brass/25 text-[10px]">Study 01</span>
            <span>·</span>
            <span className="text-ink-muted">Objective Function Optimization</span>
          </div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            Loss Weighting vs. Asymmetric Directional Penalty
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Testing whether penalizing directional inversion errors mitigates adverse execution risk.
          </p>
        </div>

        <div className="space-y-3 text-sm leading-relaxed text-ink-body font-sans">
          <p>
            Standard cross-entropy loss treats all off-diagonal classification errors identically. In electronic 
            trading, however, confusing a Stationary regime with an Up move incurs modest opportunity cost, whereas predicting an 
            Up move when price crashes Down incurs catastrophic double slippage. We formulated an asymmetric cost-sensitive loss:
          </p>

          <div className="rounded-xl border border-paper-rule bg-paper-surface/60 p-4 font-mono text-xs text-ink-primary text-center">
            {"L_total = L_CE + λ · 1_{y=Down, ŷ=Up} + λ · 1_{y=Up, ŷ=Down}  (with λ = 0.50)"}
          </div>

          <p className="text-xs text-ink-muted leading-relaxed">
            Cost-sensitive loss achieved the highest overall accuracy (<strong className="text-ink-primary numeric-value">59.22%</strong>) and 
            reduced directional flips by <strong className="text-direction-up font-semibold numeric-value">-9.7 events</strong> (from <span className="numeric-value">408.7</span> to <span className="numeric-value">399.0</span>). 
            However, mean Macro-F1 dropped from <span className="numeric-value">0.5543</span> to <strong className="text-ink-primary numeric-value">0.5533</strong> as stationary-class sensitivity was damped. 
            This confirms an operational risk trade-off, not an unalloyed model improvement.
          </p>
        </div>

        {/* Exp 1 Progressive Disclosure Seed Table */}
        <ExpandableSection
          title="Multi-Seed Verification Breakdown (Seeds 42, 123, 999)"
          summary="Per-seed accuracy, Macro-F1, inversion counts, and class-wise F1 metrics"
          badge="3 Evaluation Seeds"
        >
          <div className="overflow-x-auto rounded-xl border border-paper-rule bg-paper-surface/30">
            <table className="w-full text-left font-sans text-xs">
              <thead>
                <tr className="bg-paper-surface/80 border-b border-paper-rule text-ink-muted text-[11px] font-sans">
                  <th className="py-2.5 px-3 font-semibold">Evaluation Arm</th>
                  <th className="py-2.5 px-2 text-center">Seed</th>
                  <th className="py-2.5 px-3 text-right">Accuracy</th>
                  <th className="py-2.5 px-3 text-right">Macro-F1</th>
                  <th className="py-2.5 px-3 text-right">Inversions</th>
                  <th className="py-2.5 px-3 text-right">F1 (Down)</th>
                  <th className="py-2.5 px-3 text-right">F1 (Flat)</th>
                  <th className="py-2.5 px-3 text-right">F1 (Up)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-paper-rule/60 text-ink-body">
                {EXP1_SEEDS.map((s, idx) => (
                  <tr key={idx} className="hover:bg-paper-surface/50">
                    <td className="py-2 px-3 font-medium text-ink-primary font-sans">{s.arm}</td>
                    <td className="py-2 px-2 text-center text-ink-muted numeric-value">{s.seed}</td>
                    <td className="py-2 px-3 text-right font-semibold text-ink-primary numeric-value">{s.accuracy.toFixed(2)}%</td>
                    <td className="py-2 px-3 text-right numeric-value">{s.macroF1.toFixed(4)}</td>
                    <td className="py-2 px-3 text-right numeric-value text-ink-muted">{s.inversions}</td>
                    <td className="py-2 px-3 text-right numeric-value text-ink-muted">{s.f1Class0?.toFixed(4)}</td>
                    <td className="py-2 px-3 text-right numeric-value text-ink-muted">{s.f1Class1?.toFixed(4)}</td>
                    <td className="py-2 px-3 text-right numeric-value text-ink-muted">{s.f1Class2?.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ExpandableSection>
      </section>

      {/* 4. Study 2: Microprice Collinearity */}
      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-6">
        <div className="space-y-1">
          <div className="flex items-center space-x-2 text-xs font-sans uppercase tracking-wide text-direction-down font-semibold">
            <span className="px-2.5 py-0.5 rounded-full bg-direction-down/10 border border-direction-down/20 text-[10px]">Study 02</span>
            <span>·</span>
            <span className="text-ink-muted">Feature Collinearity Barrier</span>
          </div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            Dynamic Microprice Expansion (Δmicro) Breakdown
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Testing whether dynamic microprice deviation adds incremental predictive power over static touch depth.
          </p>
        </div>

        <div className="space-y-3 text-sm leading-relaxed text-ink-body font-sans">
          <p>
            The microprice weights top-of-book quotes by opposite-side depth: Pmicro = (Pask · Qbid + Pbid · Qask) / (Qbid + Qask). 
            We evaluated appending Δmicro = Pmicro - Pmid to the 11-feature baseline (Arm B, 12 features).
          </p>
          <p className="text-xs text-ink-muted leading-relaxed">
            Out-of-fold accuracy declined by <strong className="text-direction-down numeric-value">-58 bps</strong> (<span className="numeric-value">58.52%</span> vs <span className="numeric-value">59.10%</span>) and Macro-F1 fell to <strong className="text-ink-primary numeric-value">0.5485</strong>. 
            Collinearity diagnostics established that Δmicro is linearly coupled with depth imbalance and spread, 
            pushing Variance Inflation Factors (VIF) beyond <span className="numeric-value">10.0</span> and destabilizing recurrent weights. 
            The feature addition hypothesis was conclusively <strong className="text-direction-down">rejected</strong>.
          </p>
        </div>

        {/* Exp 2 Progressive Disclosure Seed Table */}
        <ExpandableSection
          title="Multi-Seed Verification Breakdown (Seeds 42, 123, 999)"
          summary="Per-seed comparison between 11-feature Baseline and 12-feature Microprice arm"
          badge="Collinearity Diagnostics"
        >
          <div className="overflow-x-auto rounded-xl border border-paper-rule bg-paper-surface/30">
            <table className="w-full text-left font-sans text-xs">
              <thead>
                <tr className="bg-paper-surface/80 border-b border-paper-rule text-ink-muted text-[11px] font-sans">
                  <th className="py-2.5 px-3 font-semibold">Evaluation Arm</th>
                  <th className="py-2.5 px-2 text-center">Seed</th>
                  <th className="py-2.5 px-3 text-right">Accuracy</th>
                  <th className="py-2.5 px-3 text-right">Macro-F1</th>
                  <th className="py-2.5 px-3 text-right">Inversions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-paper-rule/60 text-ink-body">
                {EXP2_SEEDS.map((s, idx) => (
                  <tr key={idx} className="hover:bg-paper-surface/50">
                    <td className="py-2 px-3 font-medium text-ink-primary font-sans">{s.arm}</td>
                    <td className="py-2 px-2 text-center text-ink-muted numeric-value">{s.seed}</td>
                    <td className="py-2 px-3 text-right font-semibold text-ink-primary numeric-value">{s.accuracy.toFixed(2)}%</td>
                    <td className="py-2 px-3 text-right numeric-value">{s.macroF1.toFixed(4)}</td>
                    <td className="py-2 px-3 text-right numeric-value text-ink-muted">{s.inversions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ExpandableSection>
      </section>

      {/* 5. Study 3: Order Flow Imbalance */}
      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-6">
        <div className="space-y-1">
          <div className="flex items-center space-x-2 text-xs font-sans uppercase tracking-wide text-direction-down font-semibold">
            <span className="px-2.5 py-0.5 rounded-full bg-direction-down/10 border border-direction-down/20 text-[10px]">Study 03</span>
            <span>·</span>
            <span className="text-ink-muted">Discrete Flow Dynamics</span>
          </div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            Quote-Level Order Flow Imbalance (OFI) Dynamics
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Evaluating discrete event flow increments versus instantaneous touch queue depth.
          </p>
        </div>

        <div className="space-y-3 text-sm leading-relaxed text-ink-body font-sans">
          <p>
            Order Flow Imbalance aggregates quote events into a scalar net flow. We evaluated two arms: Arm B (Baseline + Quote OFI, 12 features) and Arm C (OFI replacing depth imbalance, 11 features).
          </p>
          <p className="text-xs text-ink-muted leading-relaxed">
            Adding OFI decreased accuracy to <strong className="text-direction-down numeric-value">58.24%</strong> (<span className="numeric-value">-86 bps</span>). Replacing depth imbalance caused 
            an acute performance collapse to <strong className="text-direction-down numeric-value">56.32%</strong> (<span className="numeric-value">-278 bps</span>) and escalated inversions to <span className="numeric-value">452.0</span>. 
            On 1-second continuous tick samples, the instantaneous queue profile contains superior forward information 
            compared to discrete quotation changes. The OFI substitution hypothesis was conclusively <strong className="text-direction-down">rejected</strong>.
          </p>
        </div>

        {/* Exp 3 Progressive Disclosure Seed Table */}
        <ExpandableSection
          title="Multi-Seed Verification Breakdown (Seeds 42, 123, 999)"
          summary="Per-seed comparison across Baseline, OFI Addition (12D), and OFI Substitution (11D)"
          badge="Flow Increments"
        >
          <div className="overflow-x-auto rounded-xl border border-paper-rule bg-paper-surface/30">
            <table className="w-full text-left font-sans text-xs">
              <thead>
                <tr className="bg-paper-surface/80 border-b border-paper-rule text-ink-muted text-[11px] font-sans">
                  <th className="py-2.5 px-3 font-semibold">Evaluation Arm</th>
                  <th className="py-2.5 px-2 text-center">Seed</th>
                  <th className="py-2.5 px-3 text-right">Accuracy</th>
                  <th className="py-2.5 px-3 text-right">Macro-F1</th>
                  <th className="py-2.5 px-3 text-right">Inversions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-paper-rule/60 text-ink-body">
                {EXP3_SEEDS.map((s, idx) => (
                  <tr key={idx} className="hover:bg-paper-surface/50">
                    <td className="py-2 px-3 font-medium text-ink-primary font-sans">{s.arm}</td>
                    <td className="py-2 px-2 text-center text-ink-muted numeric-value">{s.seed}</td>
                    <td className="py-2 px-3 text-right font-semibold text-ink-primary numeric-value">{s.accuracy.toFixed(2)}%</td>
                    <td className="py-2 px-3 text-right numeric-value">{s.macroF1.toFixed(4)}</td>
                    <td className="py-2 px-3 text-right numeric-value text-ink-muted">{s.inversions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ExpandableSection>
      </section>
    </article>
  );
}
