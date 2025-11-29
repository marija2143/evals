#!/usr/bin/env python3
"""
Cohen's Kappa Calculator
========================
Understand why Cohen's Kappa matters more than accuracy for evaluations.

Run: uv run examples/kappa_calculator.py

What you'll learn:
- Why accuracy is misleading with imbalanced data
- How Cohen's Kappa accounts for chance agreement
- How to interpret Kappa values (Eugene Yan's targets)

No API calls - pure Python for learning the math.
"""


def calculate_metrics(predictions: list[str], ground_truth: list[str]) -> dict:
    """
    Calculate accuracy and Cohen's Kappa from predictions vs ground truth.

    Args:
        predictions: List of "pass" or "fail" from the evaluator
        ground_truth: List of "pass" or "fail" (human-verified labels)

    Returns:
        Dictionary with accuracy, kappa, and confusion matrix
    """
    assert len(predictions) == len(ground_truth), "Lists must be same length"

    # Build confusion matrix
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p == "pass" and g == "pass")
    tn = sum(1 for p, g in zip(predictions, ground_truth) if p == "fail" and g == "fail")
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p == "pass" and g == "fail")
    fn = sum(1 for p, g in zip(predictions, ground_truth) if p == "fail" and g == "pass")

    n = len(predictions)

    # Accuracy: simple % correct
    accuracy = (tp + tn) / n

    # Cohen's Kappa: agreement beyond chance
    # ----------------------------------------
    # po = observed agreement (how often we matched)
    po = (tp + tn) / n

    # pe = expected agreement if both guessed randomly
    # Based on the marginal probabilities of each class
    p_pred_pass = (tp + fp) / n  # How often predictor says pass
    p_true_pass = (tp + fn) / n  # How often ground truth is pass
    p_pred_fail = (tn + fn) / n  # How often predictor says fail
    p_true_fail = (tn + fp) / n  # How often ground truth is fail

    # Expected agreement by chance
    pe = (p_pred_pass * p_true_pass) + (p_pred_fail * p_true_fail)

    # Kappa formula: (observed - expected) / (1 - expected)
    # - Kappa = 0: No better than random guessing
    # - Kappa = 1: Perfect agreement
    # - Kappa < 0: Worse than random (systematic disagreement)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 0

    # Recall for failures (Eugene Yan: prioritize catching defects)
    fail_recall = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        "accuracy": accuracy,
        "kappa": kappa,
        "fail_recall": fail_recall,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "pe": pe,  # For understanding
    }


def interpret_kappa(kappa: float) -> str:
    """
    Interpret Kappa value per Eugene Yan's guidelines.

    Target: 0.4-0.6 (substantial agreement)
    Note: Human inter-rater reliability is often only 0.2-0.3!
    """
    if kappa < 0:
        return "Worse than chance (systematic disagreement)"
    elif kappa < 0.2:
        return "Slight agreement"
    elif kappa < 0.4:
        return "Fair agreement"
    elif kappa < 0.6:
        return "Substantial agreement (TARGET)"
    elif kappa < 0.8:
        return "Excellent agreement"
    else:
        return "Near-perfect agreement"


# =============================================================================
# Example scenarios to understand the math
# =============================================================================

SCENARIOS = {
    "Perfect agreement": {
        "predictions":   ["pass", "pass", "fail", "fail", "pass", "fail", "pass", "fail"],
        "ground_truth":  ["pass", "pass", "fail", "fail", "pass", "fail", "pass", "fail"],
    },
    "Always predicts pass (imbalanced)": {
        "predictions":   ["pass", "pass", "pass", "pass", "pass", "pass", "pass", "pass"],
        "ground_truth":  ["pass", "pass", "pass", "pass", "pass", "pass", "fail", "fail"],
    },
    "Realistic evaluator": {
        "predictions":   ["pass", "pass", "fail", "fail", "pass", "pass", "fail", "pass"],
        "ground_truth":  ["pass", "pass", "fail", "pass", "pass", "fail", "fail", "pass"],
    },
    "Random guessing": {
        "predictions":   ["pass", "fail", "pass", "fail", "pass", "fail", "pass", "fail"],
        "ground_truth":  ["pass", "pass", "fail", "fail", "fail", "pass", "pass", "fail"],
    },
}


if __name__ == "__main__":
    print("=" * 70)
    print("COHEN'S KAPPA: WHY IT MATTERS")
    print("=" * 70)
    print()
    print("The problem with accuracy:")
    print("  If 90% of samples are 'pass', always predicting 'pass' = 90% accuracy!")
    print("  But you're not actually evaluating - you're just guessing the majority.")
    print()
    print("Cohen's Kappa solves this by measuring agreement BEYOND chance.")
    print()
    print("-" * 70)

    for name, data in SCENARIOS.items():
        print(f"\nScenario: {name}")
        print(f"  Predictions:   {data['predictions']}")
        print(f"  Ground Truth:  {data['ground_truth']}")

        metrics = calculate_metrics(data["predictions"], data["ground_truth"])

        print(f"\n  Results:")
        print(f"    Accuracy:    {metrics['accuracy']:.1%}")
        print(f"    Kappa:       {metrics['kappa']:.3f} ({interpret_kappa(metrics['kappa'])})")
        print(f"    Fail Recall: {metrics['fail_recall']:.1%}")
        print(f"    Chance (pe): {metrics['pe']:.3f}")
        print(f"    Confusion:   TP={metrics['confusion_matrix']['tp']}, "
              f"TN={metrics['confusion_matrix']['tn']}, "
              f"FP={metrics['confusion_matrix']['fp']}, "
              f"FN={metrics['confusion_matrix']['fn']}")

    print()
    print("=" * 70)
    print("KEY INSIGHT: 'Always predicts pass' has 75% accuracy but Kappa = 0")
    print("  -> Accuracy is misleading! Kappa shows it's no better than chance.")
    print()
    print("Eugene Yan's targets:")
    print("  - Kappa 0.4-0.6: Substantial agreement (good)")
    print("  - Kappa > 0.6: Excellent agreement (very good)")
    print("  - Fail Recall > 80%: Catching defects is critical")
    print("=" * 70)
