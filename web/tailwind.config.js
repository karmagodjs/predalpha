/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        paper: {
          bg: "#0d0f12",       // Warm deep editorial background
          canvas: "#12151b",   // Elevated editorial reading canvas
          surface: "#171b22",  // Secondary container / table row
          hover: "#1e232c",    // Subtle interactive hover
          rule: "#242934",     // Hairline rule divider
          border: "#2e3442",   // Distinct border
        },
        ink: {
          primary: "#f3f5f8",  // Clean off-white primary text
          body: "#c4c8d2",     // High-readability body prose
          muted: "#8890a0",    // Secondary / metadata
          faint: "#575e6d",    // Captions / footnotes
        },
        brass: {
          DEFAULT: "#d99b26",  // Warm scientific accent
          bright: "#e8aa38",
          dim: "#8c6214",
          subtle: "rgba(217, 155, 38, 0.08)",
        },
        direction: {
          up: "#10b981",       // Soft emerald
          down: "#f43f5e",     // Soft rose
          flat: "#787f90",     // Neutral slate
        },
      },
      fontFamily: {
        sans: [
          "var(--font-montserrat)",
          "Montserrat",
          "-apple-system",
          "BlinkMacSystemFont",
          '"Segoe UI"',
          "Roboto",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
        numeral: [
          "var(--font-inter)",
          "-apple-system",
          "BlinkMacSystemFont",
          '"Segoe UI"',
          "Roboto",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "var(--font-mono)",
          '"JetBrains Mono"',
          '"SF Mono"',
          '"Fira Code"',
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      borderRadius: {
        none: "0",
        sm: "4px",
        md: "8px",
        lg: "12px",
        xl: "16px",
        "2xl": "20px",
        "3xl": "28px",
        full: "9999px",
      },
      lineHeight: {
        relaxed: "1.7",
      },
      keyframes: {
        pageEnter: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "none" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
      },
      animation: {
        "page-enter": "pageEnter 0.18s cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "fade-in": "fadeIn 0.15s ease-out forwards",
      },
    },
  },
  plugins: [],
};
