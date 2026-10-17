import { CAUSAL_FEATURES } from "@/lib/research-data";
import { ExpandableSection } from "@/components/ExpandableSection";

export default function MethodologyPage() {
  return (
    <article className="space-y-12">
      {/* 1. Header Hero Masthead */}
      <header className="rounded-3xl border border-paper-rule bg-paper-canvas/70 p-8 sm:p-10 relative overflow-hidden space-y-5">
        <div className="flex flex-wrap items-center gap-2 text-xs font-sans text-ink-muted">
          <span className="px-3 py-1 rounded-full bg-paper-surface border border-paper-rule text-ink-primary font-semibold text-[11px] uppercase tracking-wide">
            Econometric Whitepaper
          </span>
          <span className="px-3 py-1 rounded-full bg-paper-surface/60 border border-paper-rule/60 text-[11px]">
            Causal Feature Engineering
          </span>
          <span className="px-3 py-1 rounded-full bg-brass/10 border border-brass/25 text-brass font-medium text-[11px]">
            Replication Protocol
          </span>
        </div>

        <div className="space-y-3 max-w-4xl">
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight text-ink-primary leading-tight font-sans">
            Econometric Methodology & Replication Protocol
          </h1>
          <p className="text-base sm:text-lg text-ink-body leading-relaxed font-sans font-normal">
            Formal mathematical definitions for Level-2 order book representation, chronological purged walk-forward 
            partitioning, and deterministic replication instructions.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3 border-t border-paper-rule/60 text-xs font-sans text-ink-muted">
          <div><span className="text-ink-faint block text-[11px]">Input Dimensions</span><strong className="text-ink-primary font-medium"><span className="numeric-value">10</span> Steps × <span className="numeric-value">11</span> Features</strong></div>
          <div><span className="text-ink-faint block text-[11px]">Sampling Grid</span><strong className="text-ink-primary font-medium"><span className="numeric-value">1.0s</span> Continuous Snapshots</strong></div>
          <div><span className="text-ink-faint block text-[11px]">Normalization Scope</span><strong className="text-brass font-medium">Fit on Historical Train Only</strong></div>
        </div>
      </header>

      {/* 2. The 11 Causal Features Table Panel with Progressive Disclosure */}
      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-5">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            The 11 Causal State Features
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Strictly causal backward-looking features sampled from continuous Level-2 order book snapshots.
          </p>
        </div>

        <div className="overflow-hidden rounded-2xl border border-paper-rule bg-paper-surface/30">
          <div className="overflow-x-auto">
            <table className="w-full text-left font-sans text-xs">
              <thead>
                <tr className="border-b border-paper-rule bg-paper-surface/80 text-ink-muted text-[11px] font-sans">
                  <th className="py-3 px-4 font-semibold">Feature Identifier</th>
                  <th className="py-3 px-3 font-semibold">Category</th>
                  <th className="py-3 px-3 font-semibold">Mathematical Formulation</th>
                  <th className="py-3 px-4 font-semibold">Economic Justification</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-paper-rule/60 text-ink-body">
                {CAUSAL_FEATURES.map((f, idx) => (
                  <tr key={idx} className="hover:bg-paper-surface/50 transition-colors">
                    <td className="py-3.5 px-4 font-semibold text-ink-primary whitespace-nowrap font-sans text-xs">
                      {f.name}
                    </td>
                    <td className="py-3.5 px-3 whitespace-nowrap">
                      <span className="px-2 py-0.5 rounded-full bg-paper-canvas text-ink-muted text-[10px] border border-paper-rule font-sans">
                        {f.category}
                      </span>
                    </td>
                    <td className="py-3.5 px-3 text-brass font-medium whitespace-nowrap font-mono text-xs">
                      {f.formula}
                    </td>
                    <td className="py-3.5 px-4 text-xs font-sans text-ink-muted leading-relaxed">
                      {f.description}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* 3. Chronological Purged Walk-Forward Protocol */}
      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-6">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            Chronological Purged Partitioning Protocol
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Eliminating temporal autocorrelation leakage and forward lookahead bias.
          </p>
        </div>

        <div className="space-y-3 text-xs leading-relaxed text-ink-body font-sans max-w-3xl">
          <p>
            In standard cross-validation, observations are shuffled randomly across folds. In financial microstructure, 
            order-book states exhibit persistent serial autocorrelation across successive seconds. 
            Random shuffling inadvertently leaks future volatility regimes into historical training folds, 
            overstating out-of-sample accuracy by 10 to 20 percentage points.
          </p>
          <p>
            PredAlpha enforces a <strong className="text-ink-primary font-semibold">chronological walk-forward protocol</strong> with explicit purging buffers:
          </p>
        </div>

        {/* Structured Curved Partitioning Diagram */}
        <div className="rounded-2xl border border-paper-rule/80 bg-paper-canvas/80 p-6 space-y-3">
          <div className="flex items-center justify-between text-xs font-sans text-ink-muted border-b border-paper-rule/50 pb-2">
            <span>Chronological Time Axis (t0 → t_test)</span>
            <span className="text-[11px] text-brass font-medium">Strict Walk-Forward</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-5 gap-3 pt-2">
            {/* Step 1: Train */}
            <div className="md:col-span-2 rounded-xl border border-paper-rule bg-paper-surface/70 p-4 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-ink-primary font-sans">Historical Training</span>
                <span className="text-[10px] font-sans px-2.5 py-0.5 rounded-full bg-paper-canvas border border-paper-rule text-ink-muted">t0 → t_train</span>
              </div>
              <p className="text-[11px] text-ink-muted font-sans leading-normal">
                Continuous historical Level-2 sequences. All feature scaling parameters (means, standard deviations) fit strictly on this partition.
              </p>
            </div>

            {/* Step 2: Purge 1 */}
            <div className="rounded-xl border border-rose-950/40 bg-rose-950/15 p-4 space-y-2 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-direction-down font-sans">Warmup Purge</span>
                  <span className="text-[10px] font-sans numeric-value text-direction-down font-medium">10s</span>
                </div>
                <p className="text-[11px] text-ink-muted font-sans leading-normal mt-1">
                  Excluded from evaluation to purge 10-step sequence lookback autocorrelation.
                </p>
              </div>
              <div className="text-[10px] font-sans text-direction-down font-medium">Zero Overlap</div>
            </div>

            {/* Step 3: Val */}
            <div className="rounded-xl border border-paper-rule bg-paper-surface/70 p-4 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-ink-primary font-sans">Validation</span>
                <span className="text-[10px] font-sans px-2.5 py-0.5 rounded-full bg-paper-canvas border border-paper-rule text-ink-muted">t_val (N=<span className="numeric-value">1,428</span>)</span>
              </div>
              <p className="text-[11px] text-ink-muted font-sans leading-normal">
                Out-of-fold cohort for hyperparameter selection and diagnostic error evaluation.
              </p>
            </div>

            {/* Step 4: Sealed Holdout */}
            <div className="rounded-xl border border-brass/30 bg-brass/10 p-4 space-y-2 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-brass font-sans">Sealed Holdout</span>
                  <span className="text-[10px] font-sans text-brass">t_test</span>
                </div>
                <p className="text-[11px] text-ink-muted font-sans leading-normal mt-1">
                  Persisted as test_scaled.npz. Sealed on disk to guarantee zero p-hacking.
                </p>
              </div>
              <div className="text-[10px] font-sans text-brass font-semibold">Protected Seal</div>
            </div>
          </div>
        </div>

        <p className="text-xs text-ink-muted leading-relaxed font-sans max-w-3xl">
          Feature scaling parameters (means and standard deviations) are computed exclusively on the training partition. 
          Transforming validation or test features using statistics calculated over the entire dataset is strictly prohibited.
        </p>
      </section>

      {/* 4. Sealed Holdout Dataset Protocol with Progressive Disclosure */}
      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-4">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            Sealed Holdout Partition Disclosure
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Guaranteed zero test-set snooping or iterative parameter tuning.
          </p>
        </div>

        <p className="text-xs text-ink-body font-sans leading-relaxed">
          The final test partition is persisted as <code className="font-mono text-ink-primary bg-paper-canvas px-2 py-0.5 rounded border border-paper-rule text-xs">data/clean_v2/07_scaled/.../test_scaled.npz</code> (<span className="numeric-value">182,401</span> bytes). 
        </p>

        <ExpandableSection
          title="Sealed Holdout Protocol Rules"
          summary="Pre-declared scientific guardrails governing the final unsealed evaluation partition"
          badge="Protocol Commitment"
        >
          <ul className="space-y-2 text-xs font-sans text-ink-muted list-disc list-inside">
            <li>The test dataset was never unsealed, loaded, or evaluated during feature exploration or architecture search.</li>
            <li>Zero loss hyperparameters or model dimensions were adjusted against test fold metrics.</li>
            <li>Unsealing is permitted exclusively for final reproduction after all research hypotheses are frozen.</li>
          </ul>
        </ExpandableSection>
      </section>

      {/* 5. Deterministic Replication Instructions */}
      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-5">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            Deterministic Replication Instructions
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Reproduce all experimental metrics directly from the source repository.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-xl border border-paper-rule/70 bg-paper-surface/50 p-4 space-y-2">
            <div className="text-xs font-semibold text-ink-primary font-sans">
              Controlled Experiment 1 (Loss Weighting & Penalty):
            </div>
            <pre className="rounded-lg border border-paper-rule bg-paper-canvas p-3 text-ink-primary font-mono text-xs overflow-x-auto select-all">
python experiments/controlled_experiment_1.py
            </pre>
          </div>

          <div className="rounded-xl border border-paper-rule/70 bg-paper-surface/50 p-4 space-y-2">
            <div className="text-xs font-semibold text-ink-primary font-sans">
              Controlled Experiment 2 (Microprice Collinearity):
            </div>
            <pre className="rounded-lg border border-paper-rule bg-paper-canvas p-3 text-ink-primary font-mono text-xs overflow-x-auto select-all">
python experiments/controlled_experiment_2.py
            </pre>
          </div>

          <div className="rounded-xl border border-paper-rule/70 bg-paper-surface/50 p-4 space-y-2">
            <div className="text-xs font-semibold text-ink-primary font-sans">
              Controlled Experiment 3 (Quote-Level OFI):
            </div>
            <pre className="rounded-lg border border-paper-rule bg-paper-canvas p-3 text-ink-primary font-mono text-xs overflow-x-auto select-all">
python experiments/controlled_experiment_3.py
            </pre>
          </div>

          <div className="rounded-xl border border-paper-rule/70 bg-paper-surface/50 p-4 space-y-2">
            <div className="text-xs font-semibold text-ink-primary font-sans">
              Offline Unit and Integration Test Suite:
            </div>
            <pre className="rounded-lg border border-paper-rule bg-paper-canvas p-3 text-ink-primary font-mono text-xs overflow-x-auto select-all">
pytest -q tests/pipeline_v2/
            </pre>
          </div>
        </div>
      </section>
    </article>
  );
}
