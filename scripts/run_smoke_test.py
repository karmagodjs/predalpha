"""Command-line runner for PredAlpha-HFT Minimal Training Smoke Test.

Usage:
    python scripts/run_smoke_test.py
    python scripts/run_smoke_test.py --output-dir data/processed/phase3_smoke_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.smoke_test import (
    DEFAULT_SMOKE_CONFIG,
    SmokeTestConfig,
    run_training_smoke_test,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for training smoke test."""
    parser = argparse.ArgumentParser(
        description="Run minimal training smoke test on canonical Phase 1 dataset."
    )
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=DEFAULT_SMOKE_CONFIG.phase1_dir,
        help="Path to Phase 1 data directory containing train/val/test splits (default: data/processed/phase1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SMOKE_CONFIG.output_dir,
        help="Output directory for smoke test artifacts (default: data/processed/phase3_smoke_test).",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default=DEFAULT_SMOKE_CONFIG.class_weight,
        choices=["balanced", "none"],
        help="Class weight strategy for LogisticRegression (default: balanced).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_SMOKE_CONFIG.random_seed,
        help="Random seed for model reproducibility (default: 42).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_SMOKE_CONFIG.max_iter,
        help="Maximum iterations for solver convergence (default: 1000).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cw = None if args.class_weight == "none" else args.class_weight
    config = SmokeTestConfig(
        phase1_dir=args.phase1_dir,
        output_dir=args.output_dir,
        class_weight=cw,
        random_seed=args.random_seed,
        max_iter=args.max_iter,
    )

    print("=" * 75)
    print("PREDALPHA-HFT: TRAINING PIPELINE SMOKE TEST")
    print("=" * 75)
    print("NOTICE: This is a pipeline smoke test to verify training and inference execution.")
    print("        It does NOT provide evidence of predictive edge or profitability.")
    print("-" * 75)

    try:
        results = run_training_smoke_test(
            config=config,
            phase1_dir=args.phase1_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"\n[ERROR] Smoke test execution failed: {exc}", file=sys.stderr)
        return 1

    meta = results["smoke_test_metadata"]
    m_eval = results["evaluation_metrics"]["model"]
    d_eval = results["evaluation_metrics"]["dummy_baseline_reference"]

    print("\n1. DATASET & LABELS CONFIRMATION:")
    print(f"   Target:  {meta['target_column']} -> Classes: {meta['target_classes']}")
    print(f"   Train:   {meta['split_row_counts']['train']} rows -> {meta['class_distribution']['train']}")
    print(f"   Val:     {meta['split_row_counts']['validation']} rows -> {meta['class_distribution']['validation']}")
    print(f"   Test:    {meta['split_row_counts']['test']} rows -> {meta['class_distribution']['test']}")
    print(f"   Features ({len(meta['feature_columns'])}): {meta['feature_columns']}")

    print("\n2. EVALUATION METRICS:")
    print(f"   {'Split':12} | {'Model':22} | {'Accuracy':8} | {'Balanced Acc':12} | {'Macro F1':8}")
    print("   " + "-" * 70)
    for split_name in ["train", "validation", "test"]:
        m = m_eval[split_name]
        d = d_eval[split_name]
        print(f"   {split_name.capitalize():12} | {'LogisticRegression':22} | {m['accuracy']:8.4f} | {m['balanced_accuracy']:12.4f} | {m['macro_f1']:8.4f}")
        print(f"   {split_name.capitalize():12} | {'Dummy (Baseline)':22} | {d['accuracy']:8.4f} | {d['balanced_accuracy']:12.4f} | {d['macro_f1']:8.4f}")
        if split_name != "test":
            print("   " + "-" * 70)

    print("\n3. ARTIFACTS EXPORTED TO:")
    print(f"   {args.output_dir.resolve()}")
    print("   - smoke_test_model.joblib")
    print("   - smoke_test_metrics.json")
    print("   - smoke_test_config.json")
    print("   - confusion_matrices.json")
    print("   - smoke_test_report.md")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
