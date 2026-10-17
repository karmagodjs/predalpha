interface ConfusionMatrixProps {
  matrix: number[][]; // [Actual][Predicted]: 3x3
  title?: string;
}

export function ConfusionMatrix({ matrix, title }: ConfusionMatrixProps) {
  const rowLabels = ["Class 0: Down", "Class 1: Stationary", "Class 2: Up"];
  const colLabels = ["Pred: Down", "Pred: Stationary", "Pred: Up"];

  const rowSums = matrix.map((row) => row.reduce((a, b) => a + b, 0));
  const total = matrix.flat().reduce((a, b) => a + b, 0);
  const correct = matrix[0][0] + matrix[1][1] + matrix[2][2];
  const accuracy = total > 0 ? (correct / total) * 100 : 0;
  const inversions = matrix[0][2] + matrix[2][0];
  const inversionPct = total > 0 ? (inversions / total) * 100 : 0;

  return (
    <div className="space-y-4">
      {title && (
        <div className="flex items-baseline justify-between border-b border-paper-rule pb-2.5">
          <h4 className="text-sm font-semibold text-ink-primary font-sans">
            {title}
          </h4>
          <span className="numeric-value text-xs text-ink-faint">
            N = {total.toLocaleString()} sequences
          </span>
        </div>
      )}

      {/* Scientific Table Layout with curved container */}
      <div className="overflow-hidden rounded-2xl border border-paper-rule bg-paper-surface/30">
        <div className="overflow-x-auto">
          <table className="w-full text-left font-sans text-xs">
            <thead>
              <tr className="border-b border-paper-rule bg-paper-surface/60 text-ink-muted text-[11px]">
                <th className="py-3 px-4 font-medium font-sans">True Label \ Predicted</th>
                {colLabels.map((col, idx) => (
                  <th key={idx} className="py-3 px-3 font-medium text-center font-sans">
                    {col}
                  </th>
                ))}
                <th className="py-3 px-4 font-medium text-right font-sans">Row Total</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-paper-rule/60 text-ink-body">
              {matrix.map((row, rIdx) => {
                const rSum = rowSums[rIdx] || 1;
                return (
                  <tr key={rIdx} className="hover:bg-paper-surface/40 transition-colors">
                    <td className="py-3.5 px-4 font-medium text-ink-primary whitespace-nowrap font-sans text-xs">
                      {rowLabels[rIdx]}
                    </td>
                    {row.map((val, cIdx) => {
                      const pct = ((val / rSum) * 100).toFixed(1);
                      const isDiagonal = rIdx === cIdx;
                      const isInversion = (rIdx === 0 && cIdx === 2) || (rIdx === 2 && cIdx === 0);

                      return (
                        <td
                          key={cIdx}
                          className="py-3.5 px-3 text-center"
                        >
                          <div
                            className={`inline-block w-full py-1.5 px-2 rounded-lg transition-colors ${
                              isInversion
                                ? "text-direction-down font-bold bg-rose-950/25 border border-direction-down/20"
                                : isDiagonal
                                ? "text-ink-primary font-semibold bg-paper-surface/80 border border-paper-rule"
                                : "text-ink-muted"
                            }`}
                          >
                            <div className="numeric-value font-semibold text-xs">{val.toLocaleString()}</div>
                            <div className="text-[10px] text-ink-faint mt-0.5 numeric-value">{pct}%</div>
                          </div>
                        </td>
                      );
                    })}
                    <td className="py-3.5 px-4 text-right font-medium text-ink-muted numeric-value">
                      {rSum.toLocaleString()}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Explanatory Annotations inside curved panel */}
      <div className="rounded-xl border border-paper-rule/70 bg-paper-surface/20 p-3.5 space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-y-2 text-xs text-ink-muted font-sans">
          <div>
            <span>Overall Accuracy: </span>
            <strong className="text-ink-primary numeric-value">{accuracy.toFixed(2)}%</strong>
            <span className="text-ink-faint numeric-value"> ({correct}/{total})</span>
          </div>
          <div>
            <span>Directional Flips (Down↔Up): </span>
            <strong className="text-direction-down numeric-value">{inversions}</strong>
            <span className="text-ink-faint"> (<span className="numeric-value">{inversionPct.toFixed(1)}%</span> of cohort)</span>
          </div>
        </div>
        <p className="text-[11px] text-ink-faint leading-normal font-sans">
          * Shaded rose cells represent severe directional inversions: predicting Up when the market moves Down, or Down when the market moves Up.
        </p>
      </div>
    </div>
  );
}
