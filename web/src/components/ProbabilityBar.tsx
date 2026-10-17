interface ProbabilityBarProps {
  probabilities: {
    "Down (0)": number;
    "Stationary (1)": number;
    "Up (2)": number;
  };
  predictedLabel: "Down" | "Stationary" | "Up";
  logits?: number[];
}

export function ProbabilityBar({
  probabilities,
  predictedLabel,
  logits,
}: ProbabilityBarProps) {
  const items = [
    {
      key: "Down (0)",
      name: "Class 0 · Down",
      shortLabel: "Down",
      pct: probabilities["Down (0)"] || 0,
      logit: logits ? logits[0] : null,
      color: "bg-direction-down",
      textColor: "text-direction-down",
      isWinner: predictedLabel === "Down",
    },
    {
      key: "Stationary (1)",
      name: "Class 1 · Stationary",
      shortLabel: "Stationary",
      pct: probabilities["Stationary (1)"] || 0,
      logit: logits ? logits[1] : null,
      color: "bg-ink-muted",
      textColor: "text-ink-body",
      isWinner: predictedLabel === "Stationary",
    },
    {
      key: "Up (2)",
      name: "Class 2 · Up",
      shortLabel: "Up",
      pct: probabilities["Up (2)"] || 0,
      logit: logits ? logits[2] : null,
      color: "bg-direction-up",
      textColor: "text-direction-up",
      isWinner: predictedLabel === "Up",
    },
  ];

  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <div className="flex h-3.5 w-full overflow-hidden bg-paper-canvas rounded-full border border-paper-rule p-0.5">
          {items.map((item) => (
            <div
              key={item.key}
              className={`${item.color} transition-all duration-200 ease-out opacity-90 rounded-full`}
              style={{ width: `${Math.max(2, item.pct * 100)}%` }}
              title={`${item.name}: ${(item.pct * 100).toFixed(1)}%`}
            />
          ))}
        </div>
      </div>

      <div className="divide-y divide-paper-rule/60 border-t border-b border-paper-rule">
        {items.map((item) => (
          <div
            key={item.key}
            className={`flex items-center justify-between py-2.5 px-3 text-xs transition-colors rounded-lg my-0.5 ${
              item.isWinner ? "bg-paper-surface/60 font-semibold" : "hover:bg-paper-canvas/40"
            }`}
          >
            <div className="flex items-center space-x-3">
              <span className={`w-2.5 h-2.5 rounded-full ${item.color}`} />
              <span className={`font-sans ${item.isWinner ? "text-ink-primary font-semibold" : "text-ink-muted"}`}>
                {item.name}
              </span>
              {item.isWinner && (
                <span className="text-[10px] font-sans font-medium text-brass border border-brass/30 px-2 py-0.5 rounded-full bg-brass/10">
                  Predicted
                </span>
              )}
            </div>

            <div className="flex items-center space-x-6">
              {item.logit !== null && (
                <span className="text-ink-faint text-[11px] hidden sm:inline-block numeric-value">
                  z = {item.logit.toFixed(3)}
                </span>
              )}
              <span className={`text-sm numeric-value ${item.isWinner ? "text-ink-primary font-bold" : "text-ink-muted font-medium"}`}>
                {(item.pct * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
