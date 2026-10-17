export interface ExperimentSummaryRow {
  experiment: string;
  arm: string;
  features: number;
  accuracy: number;
  macroF1: number;
  inversions: number;
  deltaBps: number;
  finding: string;
  status: "BASELINE" | "TRADE-OFF" | "REJECTED";
}

export const CANONICAL_EXPERIMENT_SUMMARY: ExperimentSummaryRow[] = [
  {
    experiment: "Baseline Control",
    arm: "SmallLSTM (11 features)",
    features: 11,
    accuracy: 59.10,
    macroF1: 0.5543,
    inversions: 408.7,
    deltaBps: 0,
    finding: "Chronologically validated baseline (Multi-seed mean: 42, 123, 999)",
    status: "BASELINE",
  },
  {
    experiment: "Exp 1: Loss Weighting",
    arm: "Inverse-Frequency Weighting",
    features: 11,
    accuracy: 58.40,
    macroF1: 0.5512,
    inversions: 409.0,
    deltaBps: -70,
    finding: "Class weighting degraded stationary-class precision and overall accuracy",
    status: "REJECTED",
  },
  {
    experiment: "Exp 1: Loss Weighting",
    arm: "Cost-Sensitive Loss (λ=0.50)",
    features: 11,
    accuracy: 59.22,
    macroF1: 0.5533,
    inversions: 399.0,
    deltaBps: +12,
    finding: "Highest accuracy & reduced inversions (-9.7), but trade-off on minority F1",
    status: "TRADE-OFF",
  },
  {
    experiment: "Exp 2: Microprice",
    arm: "Baseline + Δ_micro",
    features: 12,
    accuracy: 58.52,
    macroF1: 0.5485,
    inversions: 412.7,
    deltaBps: -58,
    finding: "Multicollinearity (VIF > 10, condition jump); degraded generalization",
    status: "REJECTED",
  },
  {
    experiment: "Exp 3: Order Flow (OFI)",
    arm: "Baseline + Quote OFI",
    features: 12,
    accuracy: 58.24,
    macroF1: 0.5478,
    inversions: 422.3,
    deltaBps: -86,
    finding: "OFI redundant with depth imbalance on normalized tick data",
    status: "REJECTED",
  },
  {
    experiment: "Exp 3: Order Flow (OFI)",
    arm: "OFI Replacing Depth Imbalance",
    features: 11,
    accuracy: 56.32,
    macroF1: 0.5387,
    inversions: 452.0,
    deltaBps: -278,
    finding: "Severe collapse: static touch depth profile essential for 5s horizon",
    status: "REJECTED",
  },
];

export interface SeedMetricRecord {
  arm: string;
  seed: number;
  accuracy: number;
  macroF1: number;
  inversions: number;
  f1Class0?: number;
  f1Class1?: number;
  f1Class2?: number;
}

export const EXP1_SEEDS: SeedMetricRecord[] = [
  { arm: "Baseline (Cross-Entropy)", seed: 42, accuracy: 59.17, macroF1: 0.5543, inversions: 408, f1Class0: 0.6405, f1Class1: 0.3804, f1Class2: 0.6419 },
  { arm: "Baseline (Cross-Entropy)", seed: 123, accuracy: 59.10, macroF1: 0.5546, inversions: 407, f1Class0: 0.6384, f1Class1: 0.3837, f1Class2: 0.6418 },
  { arm: "Baseline (Cross-Entropy)", seed: 999, accuracy: 59.03, macroF1: 0.5539, inversions: 411, f1Class0: 0.6387, f1Class1: 0.3802, f1Class2: 0.6429 },
  { arm: "Class Weighted", seed: 42, accuracy: 58.47, macroF1: 0.5516, inversions: 408, f1Class0: 0.6341, f1Class1: 0.3807, f1Class2: 0.6399 },
  { arm: "Class Weighted", seed: 123, accuracy: 58.40, macroF1: 0.5512, inversions: 409, f1Class0: 0.6331, f1Class1: 0.3805, f1Class2: 0.6399 },
  { arm: "Class Weighted", seed: 999, accuracy: 58.33, macroF1: 0.5507, inversions: 410, f1Class0: 0.6322, f1Class1: 0.3800, f1Class2: 0.6400 },
  { arm: "Cost Sensitive (λ=0.50)", seed: 42, accuracy: 59.24, macroF1: 0.5532, inversions: 399, f1Class0: 0.6399, f1Class1: 0.3789, f1Class2: 0.6408 },
  { arm: "Cost Sensitive (λ=0.50)", seed: 123, accuracy: 59.24, macroF1: 0.5535, inversions: 398, f1Class0: 0.6402, f1Class1: 0.3792, f1Class2: 0.6411 },
  { arm: "Cost Sensitive (λ=0.50)", seed: 999, accuracy: 59.17, macroF1: 0.5531, inversions: 400, f1Class0: 0.6395, f1Class1: 0.3788, f1Class2: 0.6410 },
];

export const EXP2_SEEDS: SeedMetricRecord[] = [
  { arm: "Arm A: Baseline (11 features)", seed: 42, accuracy: 59.17, macroF1: 0.5543, inversions: 408 },
  { arm: "Arm A: Baseline (11 features)", seed: 123, accuracy: 59.10, macroF1: 0.5546, inversions: 407 },
  { arm: "Arm A: Baseline (11 features)", seed: 999, accuracy: 59.03, macroF1: 0.5539, inversions: 411 },
  { arm: "Arm B: Baseline + Δ_micro (12 features)", seed: 42, accuracy: 58.54, macroF1: 0.5488, inversions: 412 },
  { arm: "Arm B: Baseline + Δ_micro (12 features)", seed: 123, accuracy: 58.61, macroF1: 0.5489, inversions: 411 },
  { arm: "Arm B: Baseline + Δ_micro (12 features)", seed: 999, accuracy: 58.40, macroF1: 0.5477, inversions: 415 },
];

export const EXP3_SEEDS: SeedMetricRecord[] = [
  { arm: "Arm A: Baseline (11 features)", seed: 42, accuracy: 59.17, macroF1: 0.5543, inversions: 408 },
  { arm: "Arm A: Baseline (11 features)", seed: 123, accuracy: 59.10, macroF1: 0.5546, inversions: 407 },
  { arm: "Arm A: Baseline (11 features)", seed: 999, accuracy: 59.03, macroF1: 0.5539, inversions: 411 },
  { arm: "Arm B: Baseline + OFI (12 features)", seed: 42, accuracy: 58.26, macroF1: 0.5479, inversions: 422 },
  { arm: "Arm B: Baseline + OFI (12 features)", seed: 123, accuracy: 58.33, macroF1: 0.5484, inversions: 421 },
  { arm: "Arm B: Baseline + OFI (12 features)", seed: 999, accuracy: 58.12, macroF1: 0.5472, inversions: 424 },
  { arm: "Arm C: OFI replacing Depth Imbalance", seed: 42, accuracy: 56.37, macroF1: 0.5391, inversions: 451 },
  { arm: "Arm C: OFI replacing Depth Imbalance", seed: 123, accuracy: 56.44, macroF1: 0.5395, inversions: 450 },
  { arm: "Arm C: OFI replacing Depth Imbalance", seed: 999, accuracy: 56.16, macroF1: 0.5376, inversions: 455 },
];

export interface FeatureDefinition {
  name: string;
  category: "Order Book State" | "Price Dynamics" | "Quote Dynamics";
  formula: string;
  description: string;
  samplingInterval: string;
}

export const CAUSAL_FEATURES: FeatureDefinition[] = [
  {
    name: "mid_price",
    category: "Order Book State",
    formula: "(P_ask + P_bid) / 2",
    description: "Touch fair midpoint reference price at sampling timestamp",
    samplingInterval: "1 second",
  },
  {
    name: "spread",
    category: "Order Book State",
    formula: "P_ask - P_bid",
    description: "Top-of-book touch bid-ask spread width",
    samplingInterval: "1 second",
  },
  {
    name: "spread_bps",
    category: "Order Book State",
    formula: "(spread / mid_price) × 10,000",
    description: "Touch spread normalized in basis points",
    samplingInterval: "1 second",
  },
  {
    name: "microprice",
    category: "Order Book State",
    formula: "(V_ask × P_bid + V_bid × P_ask) / (V_bid + V_ask)",
    description: "Top-of-book volume-weighted microprice",
    samplingInterval: "1 second",
  },
  {
    name: "depth_imbalance",
    category: "Order Book State",
    formula: "(V_bid - V_ask) / (V_bid + V_ask)",
    description: "Normalized touch liquidity queue imbalance (-1.0 to +1.0)",
    samplingInterval: "1 second",
  },
  {
    name: "mid_return_1s",
    category: "Price Dynamics",
    formula: "ln(mid_t / mid_{t-1s})",
    description: "1-second backward-looking logarithmic mid-price return",
    samplingInterval: "1 second",
  },
  {
    name: "mid_return_3s",
    category: "Price Dynamics",
    formula: "ln(mid_t / mid_{t-3s})",
    description: "3-second backward-looking logarithmic mid-price return",
    samplingInterval: "1 second",
  },
  {
    name: "mid_return_5s",
    category: "Price Dynamics",
    formula: "ln(mid_t / mid_{t-5s})",
    description: "5-second backward-looking logarithmic mid-price return",
    samplingInterval: "1 second",
  },
  {
    name: "mid_volatility_5s",
    category: "Price Dynamics",
    formula: "StdDev(mid_returns_{t-5s..t})",
    description: "5-second rolling realized standard deviation of mid returns",
    samplingInterval: "1 second",
  },
  {
    name: "bid_change_1s",
    category: "Quote Dynamics",
    formula: "P_bid,t - P_bid,t-1s",
    description: "1-second top-of-book best bid quote velocity",
    samplingInterval: "1 second",
  },
  {
    name: "ask_change_1s",
    category: "Quote Dynamics",
    formula: "P_ask,t - P_ask,t-1s",
    description: "1-second top-of-book best ask quote velocity",
    samplingInterval: "1 second",
  },
];

export interface ConfidenceTier {
  threshold: number;
  coveragePct: number;
  accuracyPct: number | null;
}

export const CONFIDENCE_ANALYSIS_TIERS: ConfidenceTier[] = [
  { threshold: 0.4, coveragePct: 97.5, accuracyPct: 59.6 },
  { threshold: 0.5, coveragePct: 66.0, accuracyPct: 62.7 },
  { threshold: 0.6, coveragePct: 23.9, accuracyPct: 64.5 },
  { threshold: 0.7, coveragePct: 5.4, accuracyPct: 72.7 },
  { threshold: 0.8, coveragePct: 1.0, accuracyPct: 85.7 },
  { threshold: 0.9, coveragePct: 0.0, accuracyPct: null },
];

export interface ConfusionMatrixData {
  seed: number;
  name: string;
  matrix: number[][];
}

export const CONFUSION_MATRICES: ConfusionMatrixData[] = [
  {
    seed: 42,
    name: "SmallLSTM (Seed 42)",
    matrix: [
      [374, 27, 215],
      [57, 72, 66],
      [193, 25, 399],
    ],
  },
  {
    seed: 123,
    name: "SmallLSTM (Seed 123)",
    matrix: [
      [373, 28, 215],
      [58, 73, 64],
      [192, 26, 399],
    ],
  },
  {
    seed: 999,
    name: "SmallLSTM (Seed 999)",
    matrix: [
      [372, 27, 217],
      [58, 72, 65],
      [194, 25, 398],
    ],
  },
  {
    seed: 42,
    name: "SmallGRU (Seed 42)",
    matrix: [
      [370, 29, 217],
      [59, 70, 66],
      [195, 27, 395],
    ],
  },
];

export const VALIDATION_COHORT_META = {
  totalSequences: 1428,
  sequenceShape: [1428, 10, 11],
  downCount: 616,
  flatCount: 195,
  upCount: 617,
  meanConfidence: 0.544,
  medianConfidence: 0.531,
  confidenceP10: 0.447,
  confidenceP90: 0.658,
};
