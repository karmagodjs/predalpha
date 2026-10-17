import { CAUSAL_FEATURES } from "@/lib/research-data";
import { ExpandableSection } from "@/components/ExpandableSection";

export default function MethodologyPage() {
  return (
    <article className="space-y-12">
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
            producing spuriously high backtested accuracies that collapse in live execution.
          </p>
          <p>
            PredAlpha enforces a strict chronological partition architecture with non-overlapping evaluation windows and purged warmup gaps:
          </p>
        </div>

        <div className="rounded-2xl border border-paper-rule bg-paper-surface/30 p-5 space-y-4">
          <div className="text-xs font-semibold text-ink-primary font-sans">
            Chronological Split Sequence Partitioning
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs font-sans">
            <div className="rounded-xl border border-paper-rule/70 bg-paper-canvas/60 p-4 space-y-1.5">
              <span className="text-[10px] text-ink-faint font-semibold uppercase">Step 01</span>
              <strong className="block text-ink-primary">Training Partition</strong>
              <span className="text-xs text-ink-muted numeric-value">N = 2,966 sequences</span>
              <p className="text-[11px] text-ink-faint leading-tight pt-1">
                Historical fitting partition; scalers and architectures parameterize exclusively here.
              </p>
            </div>

            <div className="rounded-xl border border-brass/25 bg-brass/5 p-4 space-y-1.5">
              <span className="text-[10px] text-brass font-semibold uppercase">Step 02</span>
              <strong className="block text-brass">Purge Gap 1</strong>
              <span className="text-xs text-brass font-medium numeric-value">10 Seconds (10 steps)</span>
              <p className="text-[11px] text-ink-muted leading-tight pt-1">
                Zero forward overlap: full sequence receptive field purged between Train and Val.
              </p>
            </div>

            <div className="rounded-xl border border-paper-rule/70 bg-paper-canvas/60 p-4 space-y-1.5">
              <span className="text-[10px] text-ink-faint font-semibold uppercase">Step 03</span>
              <strong className="block text-ink-primary">Validation Partition</strong>
              <span className="text-xs text-direction-up font-semibold numeric-value">N = 1,428 sequences</span>
              <p className="text-[11px] text-ink-faint leading-tight pt-1">
                Model checkpoint selection, confidence threshold tuning, and error matrix analysis.
              </p>
            </div>

            <div className="rounded-xl border border-brass/25 bg-brass/5 p-4 space-y-1.5">
              <span className="text-[10px] text-brass font-semibold uppercase">Step 04</span>
              <strong className="block text-brass">Purge Gap 2 & Sealed Holdout</strong>
              <span className="text-xs text-brass font-medium numeric-value">N = 1,465 sequences</span>
              <p className="text-[11px] text-ink-muted leading-tight pt-1">
                Sealed final holdout dataset (<code className="font-mono text-ink-primary bg-paper-surface px-1 py-0.5 rounded text-[10px]">test_scaled.npz</code>) untouched on disk.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-6">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            Sealed Holdout Dataset & P-Hacking Prevention
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Pre-commitment to sealed out-of-sample data prevents exploratory data leakage.
          </p>
        </div>

        <p className="text-xs text-ink-body leading-relaxed font-sans max-w-3xl">
          A fundamental challenge in empirical finance is researcher degrees of freedom: iteratively adjusting hyperparameters 
          against a validation set until an anomaly appears by chance. To guard against this, the final test partition 
          is cryptographically sealed on disk:
        </p>

        <ExpandableSection
          title="Protocol: Sealed Holdout Integrity"
          summary="Why test_scaled.npz is never evaluated during ongoing experimentation"
          badge="Scientific Integrity"
        >
          <div className="space-y-3 text-xs leading-relaxed text-ink-body font-sans">
            <p>
              The file <code className="font-mono text-ink-primary bg-paper-canvas px-1.5 py-0.5 rounded border border-paper-rule">data/clean_v2/07_scaled/expanded_collection/test_scaled.npz</code> contains 
              <span className="numeric-value"> 1,465</span> consecutive out-of-sample sequences. In accordance with Phase 27 research protocols:
            </p>
            <ul className="space-y-1 text-ink-muted list-disc list-inside bg-paper-surface/40 p-3 rounded-lg border border-paper-rule/60">
              <li>No script in the training or experimentation pipeline evaluates against this partition.</li>
              <li>Hyperparameters, loss weights, and feature architectures are selected exclusively on validation.</li>
              <li>The holdout dataset will only be evaluated once all candidate models are permanently frozen.</li>
            </ul>
          </div>
        </ExpandableSection>
      </section>

      <section className="rounded-2xl border border-paper-rule bg-paper-canvas/50 p-6 sm:p-8 space-y-6">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-ink-primary font-sans">
            Deterministic Replication Instructions
          </h2>
          <p className="text-xs text-ink-muted font-sans">
            Reproduce baseline metrics and controlled experiments from scratch.
          </p>
        </div>

        <div className="space-y-4 max-w-3xl text-xs font-sans text-ink-body">
          <p>
            All experiments are fully deterministic when seeded with random seeds 42, 123, and 999. 
            To replicate results from the repository root:
          </p>

          <div className="space-y-3">
            <div>
              <span className="text-[11px] text-ink-muted block font-medium mb-1">1. Run Baseline Control Experiment (Multi-Seed SmallLSTM)</span>
              <pre className="rounded-xl border border-paper-rule bg-paper-surface/60 p-3 font-mono text-xs text-ink-primary overflow-x-auto">
                python experiments/baseline.py --seeds 42,123,999 --device cpu
              </pre>
            </div>

            <div>
              <span className="text-[11px] text-ink-muted block font-medium mb-1">2. Run Controlled Experiment 1 (Loss Function Ablation)</span>
              <pre className="rounded-xl border border-paper-rule bg-paper-surface/60 p-3 font-mono text-xs text-ink-primary overflow-x-auto">
                python experiments/controlled_experiment_1.py --seeds 42,123,999
              </pre>
            </div>

            <div>
              <span className="text-[11px] text-ink-muted block font-medium mb-1">3. Run Controlled Experiment 2 (Microprice Feature Addition)</span>
              <pre className="rounded-xl border border-paper-rule bg-paper-surface/60 p-3 font-mono text-xs text-ink-primary overflow-x-auto">
                python experiments/controlled_experiment_2.py --seeds 42,123,999
              </pre>
            </div>

            <div>
              <span className="text-[11px] text-ink-muted block font-medium mb-1">4. Run Controlled Experiment 3 (Order Flow Imbalance OFI)</span>
              <pre className="rounded-xl border border-paper-rule bg-paper-surface/60 p-3 font-mono text-xs text-ink-primary overflow-x-auto">
                python experiments/controlled_experiment_3.py --seeds 42,123,999
              </pre>
            </div>
          </div>
        </div>
      </section>
    </article>
  );
}
