"use client";

import { useState, ReactNode } from "react";

interface ExpandableSectionProps {
  title: string;
  summary?: string;
  children: ReactNode;
  defaultOpen?: boolean;
  badge?: string;
}

export function ExpandableSection({
  title,
  summary,
  children,
  defaultOpen = false,
  badge,
}: ExpandableSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="rounded-xl border border-paper-rule bg-paper-canvas/70 transition-colors overflow-hidden">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between p-4 text-left font-sans transition-colors hover:bg-paper-surface/50"
        aria-expanded={isOpen}
      >
        <div className="space-y-0.5 pr-4">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-ink-primary font-sans">{title}</span>
            {badge && (
              <span className="px-2 py-0.5 rounded text-[10px] font-sans font-medium bg-paper-surface text-ink-muted border border-paper-rule">
                {badge}
              </span>
            )}
          </div>
          {summary && <p className="text-[11px] text-ink-muted font-sans leading-normal">{summary}</p>}
        </div>
        <span className="text-xs font-sans text-ink-faint hover:text-ink-primary shrink-0 transition-colors">
          {isOpen ? "Hide details" : "Show details"}
        </span>
      </button>
      {isOpen && (
        <div className="px-4 pb-4 pt-2 text-xs text-ink-body font-sans border-t border-paper-rule/60 animate-fade-in space-y-3">
          {children}
        </div>
      )}
    </div>
  );
}
