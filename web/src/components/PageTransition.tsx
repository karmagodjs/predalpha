"use client";

import { usePathname } from "next/navigation";
import { ReactNode, useEffect } from "react";

interface PageTransitionProps {
  children: ReactNode;
}

export function PageTransition({ children }: PageTransitionProps) {
  const pathname = usePathname();

  useEffect(() => {
    if (!window.location.hash) {
      window.scrollTo(0, 0);
    }
  }, [pathname]);

  return (
    <div
      key={pathname}
      className="w-full animate-page-enter motion-reduce:animate-none motion-reduce:transform-none motion-reduce:opacity-100"
      data-page-transition
    >
      {children}
    </div>
  );
}
