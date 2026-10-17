"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect, useRef } from "react";

const NAV_LINKS = [
  { href: "/", label: "Overview" },
  { href: "/lab", label: "Model Lab" },
  { href: "/experiments", label: "Experiments" },
  { href: "/validation", label: "Validation" },
  { href: "/methodology", label: "Methodology" },
  { href: "/limitations", label: "Limitations" },
];

export function Navbar() {
  const pathname = usePathname();
  const [pendingHref, setPendingHref] = useState<string | null>(null);
  const navRef = useRef<HTMLElement | null>(null);
  const activeLinkRef = useRef<HTMLAnchorElement | null>(null);

  useEffect(() => {
    setPendingHref(null);
  }, [pathname]);

  useEffect(() => {
    if (navRef.current && activeLinkRef.current) {
      const nav = navRef.current;
      const link = activeLinkRef.current;
      const navRect = nav.getBoundingClientRect();
      const linkRect = link.getBoundingClientRect();
      if (linkRect.left < navRect.left || linkRect.right > navRect.right) {
        nav.scrollTo({
          left: link.offsetLeft - nav.offsetWidth / 2 + link.offsetWidth / 2,
          behavior: "smooth",
        });
      }
    }
  }, [pathname]);

  const activeHref = pendingHref ?? pathname;

  return (
    <header className="sticky top-0 z-50 w-full border-b border-paper-rule bg-paper-bg/95 backdrop-blur-sm">
      <div className="mx-auto flex h-13 max-w-6xl items-center justify-between px-4 sm:px-6">
        <div className="hidden md:block w-9 shrink-0" aria-hidden="true" />

        <nav
          ref={navRef}
          className="flex items-center space-x-1.5 sm:space-x-2 md:space-x-3 lg:space-x-5 overflow-x-auto no-scrollbar py-1 flex-1 md:flex-initial justify-start md:justify-center min-w-0"
          aria-label="Main Navigation"
        >
          {NAV_LINKS.map((link) => {
            const isActive = activeHref === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                prefetch={true}
                ref={isActive ? activeLinkRef : undefined}
                onClick={() => setPendingHref(link.href)}
                className={`whitespace-nowrap px-3 py-1.5 text-xs font-sans rounded-lg border transition-colors shrink-0 ${
                  isActive
                    ? "text-ink-primary font-semibold bg-paper-surface border-paper-rule"
                    : "text-ink-muted hover:text-ink-primary hover:bg-paper-surface/50 border-transparent"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center shrink-0 pl-2.5 sm:pl-3 md:pl-0 w-9 md:w-9 justify-end border-l border-paper-rule/50 md:border-l-0 ml-1.5 md:ml-0">
          <a
            href="https://github.com/karmagodjs/predalpha"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="View source on GitHub"
            title="View source on GitHub"
            className="p-1.5 text-ink-muted hover:text-ink-primary transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-brass rounded"
          >
            <svg
              viewBox="0 0 16 16"
              width="16"
              height="16"
              fill="currentColor"
              aria-hidden="true"
              className="w-4 h-4 shrink-0"
            >
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
            </svg>
            <span className="sr-only">View source on GitHub</span>
          </a>
        </div>
      </div>
    </header>
  );
}
