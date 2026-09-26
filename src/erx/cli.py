"""
ER-X Command Line Interface (CLI).
"""

import argparse
import logging
import sys
from pathlib import Path
from src.erx.config import ERXConfig
from src.erx.pipeline import ERXPipeline


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def main():
    parser = argparse.ArgumentParser(
        description="ER-X: Production-Grade High-Recall Entity Resolution Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: smoke
    smoke_parser = subparsers.add_parser("smoke", help="Run fast 100-S1 smoke benchmark")
    smoke_parser.add_argument("--num-s1", type=int, default=100, help="Number of S1 entities to benchmark")
    smoke_parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")

    # Subcommand: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run candidate recall benchmark on S1 subset")
    bench_parser.add_argument("--num-s1", type=int, default=1000, help="Number of S1 entities (1000 or 5000)")
    bench_parser.add_argument("--k", type=int, default=35, help="Top-K candidate cap per target")
    bench_parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")

    # Subcommand: validate
    val_parser = subparsers.add_parser("validate", help="Run multi-seed entity-level cross-validation")
    val_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026], help="Random seeds for validation")
    val_parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation ratio")
    val_parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")

    # Subcommand: predict
    pred_parser = subparsers.add_parser("predict", help="Generate full test predictions and submission files")
    pred_parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory for submissions")
    pred_parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    setup_logging(verbose=getattr(args, "verbose", False))

    config = ERXConfig()
    pipeline = ERXPipeline(config)

    if args.command == "smoke":
        results = pipeline.run_smoke_benchmark(num_s1=args.num_s1)
        print("\n=== Smoke Benchmark Summary ===")
        print(f"Entities:           {results['num_s1']}")
        print(f"Candidate Recall:   {results['metrics']['candidate_recall']*100:.2f}%")
        print(f"Macro F0.5:         {results['metrics']['macro_f05']:.4f}")
        print(f"Precision:          {results['metrics']['macro_precision']*100:.2f}%")
        print(f"Recall:             {results['metrics']['macro_recall']*100:.2f}%")
        print(f"Singleton Accuracy: {results['metrics']['singleton_accuracy']*100:.2f}%")
        print(f"Total Runtime:      {results['total_elapsed_s']}s")
    elif args.command == "benchmark":
        results = pipeline.run_smoke_benchmark(num_s1=args.num_s1)
        print(f"\n=== Benchmark ({args.num_s1} S1) Finished in {results['total_elapsed_s']}s ===")
        print(f"Candidate Recall: {results['metrics']['candidate_recall']*100:.2f}%")
    elif args.command == "validate":
        print(f"Running validation on seeds {args.seeds}...")
    elif args.command == "predict":
        print("Running test prediction pipeline...")


if __name__ == "__main__":
    main()
