#!/usr/bin/env python3
"""
Verify LLM Evaluator Against Ground Truth Labels
=================================================

This script implements Eugene Yan's eval verification methodology:
https://eugeneyan.com/writing/product-evals/

Setup:
- Ground truth: GPT-5.1 labels in data/questions_version_2.csv
- Evaluator: Configurable model (default: gpt-5-mini)
- Goal: Measure how well smaller models replicate GPT-5.1's decisions

Key Metrics (from Eugene Yan):
- Cohen's Kappa: 0.4-0.6 = substantial agreement (target)
- Prioritize RECALL on failures (catching defects is critical)
- Human inter-rater reliability is often only 0.2-0.3

Usage:
    uv run verify_evaluator.py [--sample N] [--model MODEL] [--category CAT]

Examples:
    uv run verify_evaluator.py --sample 50
    uv run verify_evaluator.py --model gpt-5-nano-2025-08-07
    uv run verify_evaluator.py --category precise_calculations

See:
    - docs/tutorial/ for learning progression
    - docs/glossary.md for terminology
    - examples/ for simple examples
"""

import random
import argparse

from evaluators.runner import (
    load_dataset,
    run_evaluation,
    print_results,
    save_run,
)

# Configuration
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
DATA_FILE = "data/questions_version_2.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Verify LLM evaluator against ground truth labels"
    )
    parser.add_argument("--sample", type=int, help="Evaluate only N random samples")
    parser.add_argument("--train-test", action="store_true", help="Use 75/25 train/test split")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use as evaluator (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/runs.jsonl",
        help="Output file for run logs (default: results/runs.jsonl)",
    )
    args = parser.parse_args()

    # Header
    print("=" * 70)
    print("LLM Evaluator Verification")
    print("=" * 70)
    print(f"\nEvaluator Model: {args.model}")
    print(f"Ground Truth: GPT-5.1 labels from {DATA_FILE}")
    print(f"Methodology: Eugene Yan's Product Evals")
    print()

    # Load data
    data = load_dataset(DATA_FILE)
    print(f"Loaded {len(data)} labeled samples")

    # Filter by category if specified
    if args.category:
        data = [d for d in data if d.get("category") == args.category]
        print(f"Filtered to {len(data)} samples in category '{args.category}'")

    # Show label distribution
    pass_count = sum(1 for d in data if d["label"] == "pass")
    fail_count = sum(1 for d in data if d["label"] == "fail")
    print(
        f"Label distribution: {pass_count} pass, {fail_count} fail "
        f"({fail_count/len(data)*100:.1f}% failure rate)"
    )

    # Sample if requested
    if args.sample:
        random.seed(42)
        data = random.sample(data, min(args.sample, len(data)))
        print(f"Sampled {len(data)} items for evaluation")

    # Train/test split if requested
    if args.train_test:
        random.seed(42)
        random.shuffle(data)
        split = int(len(data) * 0.75)
        train_data, test_data = data[:split], data[split:]
        print(f"Split: {len(train_data)} train (for prompt tuning), {len(test_data)} test")
        data = test_data

    print(f"\nEvaluating {len(data)} samples...")
    print("-" * 70)

    # Progress callback
    def progress(current, total, accuracy):
        print(f"  Progress: {current}/{total} | Accuracy so far: {accuracy*100:.1f}%", flush=True)

    # Run evaluation
    results = run_evaluation(data, args.model, progress_callback=progress)

    # Print results
    print_results(results, args.model, DATA_FILE)

    # Save run
    save_run(
        results,
        args.model,
        DATA_FILE,
        output_file=args.output,
        category_filter=args.category,
    )


if __name__ == "__main__":
    main()
