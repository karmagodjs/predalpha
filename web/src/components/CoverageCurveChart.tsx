"use client";

import { CONFIDENCE_ANALYSIS_TIERS } from "@/lib/research-data";

export function CoverageCurveChart() {
  const tiers = CONFIDENCE_ANALYSIS_TIERS;

  // Chart dimensions
  const width = 640;
  const height = 240;
  const padding = { top: 20, right: 30, bottom: 35, left: 45 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const getX = (t: number) => padding.left + ((t - 0.4) / (0.9 - 0.4)) * chartW;
  const getY = (val: number) => padding.top + chartH - (val / 100) * chartH;

  const coveragePoints = tiers.map((d) => `${getX(d.threshold)},${getY(d.coveragePct)}`).join(" ");
  const accuracyTiers = tiers.filter((d) => d.accuracyPct !== null);
  const accuracyPoints = accuracyTiers.map((d) => `${getX(d.threshold)},${getY(d.accuracyPct!)}`).join(" ");

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-paper-rule pb-3">
        <h4 className="text-sm font-semibold text-ink-primary font-sans">
          Selective Classification: Coverage vs. Accuracy Curve
        </h4>
        <div className="flex items-center space-x-5 text-xs font-sans">
          <div className="flex items-center space-x-2">
            <span className="h-2 w-4 rounded-full bg-brass"></span>
            <span className="text-ink-muted">Accuracy on covered</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="h-2 w-4 rounded-full bg-ink-muted"></span>
            <span className="text-ink-muted">Decision coverage</span>
          </div>
        </div>
      </div>

      {/* SVG Chart with curved container */}
      <div className="rounded-2xl border border-paper-rule bg-paper-surface/30 p-4 sm:p-6 overflow-hidden">
        <div className="relative w-full">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-auto text-ink-faint select-none"
          >

            {/* Grid lines */}
            {[0, 25, 50, 75, 100].map((tick) => {
              const y = getY(tick);
              return (
                <g key={tick}>
                  <line
                    x1={padding.left}
                    y1={y}
                    x2={width - padding.right}
                    y2={y}
                    stroke="#242934"
                    strokeDasharray="2,3"
                  />
                  <text
                    x={padding.left - 8}
                    y={y + 3}
                    textAnchor="end"
                    className="fill-ink-faint text-[10px]"
                    style={{ fontFamily: "var(--font-inter), -apple-system, sans-serif" }}
                  >
                    {tick}%
                  </text>
                </g>
              );
            })}

            {/* X Axis */}
            {tiers.map((d) => {
              const x = getX(d.threshold);
              return (
                <g key={d.threshold}>
                  <line
                    x1={x}
                    y1={height - padding.bottom}
                    x2={x}
                    y2={height - padding.bottom + 4}
                    stroke="#38404e"
                  />
                  <text
                    x={x}
                    y={height - padding.bottom + 18}
                    textAnchor="middle"
                    className="fill-ink-muted text-[11px]"
                    style={{ fontFamily: "var(--font-inter), -apple-system, sans-serif" }}
                  >
                    τ ≥ {d.threshold.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Coverage Polyline */}
            <polyline
              fill="none"
              stroke="#8890a0"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={coveragePoints}
            />
            {tiers.map((d) => (
              <circle
                key={d.threshold}
                cx={getX(d.threshold)}
                cy={getY(d.coveragePct)}
                r="3"
                fill="#0d0f12"
                stroke="#8890a0"
                strokeWidth="1.5"
              />
            ))}

            {/* Accuracy Polyline */}
            <polyline
              fill="none"
              stroke="#d99b26"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={accuracyPoints}
            />
            {accuracyTiers.map((d) => (
              <circle
                key={d.threshold}
                cx={getX(d.threshold)}
                cy={getY(d.accuracyPct!)}
                r="3.5"
                fill="#0d0f12"
                stroke="#d99b26"
                strokeWidth="2"
              />
            ))}
          </svg>
        </div>
      </div>

      {/* Accompanying Booktabs Table in curved container */}
      <div className="overflow-hidden rounded-2xl border border-paper-rule bg-paper-surface/30">
        <div className="overflow-x-auto">
          <table className="w-full text-left font-sans text-xs">
            <thead>
              <tr className="border-b border-paper-rule bg-paper-surface/60 text-[11px] text-ink-muted font-sans">
                <th className="py-3 px-4 font-medium">Threshold (τ)</th>
                <th className="py-3 px-4 font-medium text-right">Decision Coverage (%)</th>
                <th className="py-3 px-4 font-medium text-right">Accuracy on Covered (%)</th>
                <th className="py-3 px-4 font-medium text-right">Active Trade Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-paper-rule/60 text-ink-body">
              {tiers.map((d) => (
                <tr key={d.threshold} className="hover:bg-paper-surface/40 transition-colors">
                  <td className="py-3 px-4 font-semibold text-ink-primary numeric-value">τ ≥ {d.threshold.toFixed(1)}</td>
                  <td className="py-3 px-4 text-right numeric-value">{d.coveragePct.toFixed(1)}%</td>
                  <td className="py-3 px-4 text-right font-semibold text-brass numeric-value">
                    {d.accuracyPct !== null ? `${d.accuracyPct.toFixed(1)}%` : "—"}
                  </td>
                  <td className="py-3 px-4 text-right text-ink-muted text-[11px] font-sans">
                    {d.coveragePct > 50
                      ? "~1 per 1.5s"
                      : d.coveragePct > 20
                      ? "~1 per 4s"
                      : d.coveragePct > 5
                      ? "~1 per 18s"
                      : d.coveragePct > 0
                      ? "~1 per 100s"
                      : "Zero trades"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
