import Link from "next/link";

export function Footer() {
  return (
    <footer className="w-full border-t border-paper-rule bg-paper-bg py-8 sm:py-10 mt-14 sm:mt-16">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
          {/* Col 1: Wordmark & Provenance */}
          <div className="space-y-2 md:col-span-2">
            <div className="font-sans text-xs font-bold tracking-tight text-ink-primary">
              PREDALPHA
            </div>
            <p className="max-w-md text-[11px] leading-relaxed text-ink-muted font-sans">
              Independent quantitative research on high-frequency cryptocurrency order book microstructure 
              and deep recurrent neural networks for <span className="numeric-value">5,000 ms</span> mid-price directional forecasting.
            </p>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-ink-faint">
              <span>Spec: <span className="numeric-value">2.0</span>_Phase27_Frozen</span>
              <span>·</span>
              <span>Commit: 3de3b16</span>
              <span>·</span>
              <span>N = <span className="numeric-value">1,428</span> holdout</span>
            </div>
          </div>

          {/* Col 2: Research Sections */}
          <div>
            <div className="font-sans text-[11px] font-semibold text-ink-primary uppercase tracking-wide mb-2.5">
              Research Sections
            </div>
            <ul className="space-y-1.5 text-[11px] text-ink-muted font-sans">
              <li>
                <Link href="/" className="hover:text-ink-primary transition-colors">
                  Overview & Findings
                </Link>
              </li>
              <li>
                <Link href="/lab" className="hover:text-ink-primary transition-colors">
                  Model Lab
                </Link>
              </li>
              <li>
                <Link href="/experiments" className="hover:text-ink-primary transition-colors">
                  Controlled Experiments
                </Link>
              </li>
              <li>
                <Link href="/validation" className="hover:text-ink-primary transition-colors">
                  Validation Diagnostics
                </Link>
              </li>
              <li>
                <Link href="/methodology" className="hover:text-ink-primary transition-colors">
                  Methodology & Protocols
                </Link>
              </li>
              <li>
                <Link href="/limitations" className="hover:text-ink-primary transition-colors">
                  Execution Realities
                </Link>
              </li>
            </ul>
          </div>

          {/* Col 3: Research Invariants */}
          <div>
            <div className="font-sans text-[11px] font-semibold text-ink-primary uppercase tracking-wide mb-2.5">
              Governance & Verification
            </div>
            <ul className="space-y-1 text-[11px] text-ink-muted font-sans">
              <li>Status: Software-Release-Ready</li>
              <li>Live Orders: Unauthorized</li>
              <li>Test Dataset: Sealed & Frozen</li>
              <li>Architecture: SmallLSTM (<span className="numeric-value">5,859</span> params)</li>
              <li>Seeds: <span className="numeric-value">42</span>, <span className="numeric-value">123</span>, <span className="numeric-value">999</span></li>
            </ul>
          </div>
        </div>

        {/* Disclaimer Bar */}
        <div className="mt-8 border-t border-paper-rule/60 pt-4">
          <p className="text-[10px] leading-normal text-ink-faint font-sans">
            DISCLAIMER: All experimental evaluations, validation metrics, and simulated executions are intended exclusively 
            for academic and quantitative research. Historical validation performance does not establish live market trading profitability. 
            Live trading and prospective Session 001 execution remain strictly unauthorized.
          </p>
        </div>
      </div>
    </footer>
  );
}
