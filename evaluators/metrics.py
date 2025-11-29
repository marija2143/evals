"""
Evaluation Metrics
==================

Functions for calculating and interpreting evaluation metrics.

Key metrics (per Eugene Yan):
    - Cohen's Kappa: Agreement beyond chance (target: 0.4-0.6)
    - Fail Recall: % of failures caught (target: >80%)
    - Accuracy: Simple % correct (misleading with imbalanced data!)

See:
    - docs/glossary.md for terminology
    - docs/tutorial/05_metrics_explained.md for detailed walkthrough
    - examples/kappa_calculator.py for interactive examples
"""

import math
from typing import Union


def calculate_metrics(results: list[dict]) -> dict:
    """
    Calculate comprehensive evaluation metrics.

    Implements Eugene Yan's recommended metrics:
    https://eugeneyan.com/writing/product-evals/

    Args:
        results: List of dicts with keys:
            - "label": Ground truth ("pass" or "fail")
            - "pred": Prediction ("pass" or "fail")

    Returns:
        Dictionary containing:
            - total: Number of samples
            - tp, fp, fn, tn: Confusion matrix values
            - accuracy: Simple accuracy (use with caution!)
            - accuracy_ci: 95% confidence interval as (lower, upper)
            - precision_pass, recall_pass, f1_pass: Pass detection metrics
            - precision_fail, recall_fail, f1_fail: Fail detection metrics
            - kappa: Cohen's Kappa (the key metric!)

    Example:
        results = [
            {"label": "pass", "pred": "pass"},
            {"label": "fail", "pred": "fail"},
            {"label": "pass", "pred": "fail"},  # False negative
        ]
        metrics = calculate_metrics(results)
        print(f"Kappa: {metrics['kappa']:.3f}")
    """
    # =========================================================================
    # STEP 1: Build Confusion Matrix
    # =========================================================================
    # Confusion matrix terminology:
    #                   Actual
    #               Pass    Fail
    # Predicted
    #    Pass       TP      FP (false alarm)
    #    Fail       FN      TN (correct reject)
    #
    # See docs/glossary.md#confusion-matrix

    tp = sum(1 for r in results if r["label"] == "pass" and r["pred"] == "pass")
    fp = sum(1 for r in results if r["label"] == "fail" and r["pred"] == "pass")
    fn = sum(1 for r in results if r["label"] == "pass" and r["pred"] == "fail")
    tn = sum(1 for r in results if r["label"] == "fail" and r["pred"] == "fail")

    total = len(results)

    # =========================================================================
    # STEP 2: Basic Accuracy
    # =========================================================================
    # WARNING: Accuracy is misleading with imbalanced data!
    # See docs/common_mistakes.md#3-using-accuracy-instead-of-kappa
    accuracy = (tp + tn) / total if total > 0 else 0

    # =========================================================================
    # STEP 3: Precision/Recall for PASS
    # =========================================================================
    # Precision: "When I say pass, am I right?"
    # Recall: "Did I catch all the passes?"
    precision_pass = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pass = tp / (tp + fn) if (tp + fn) > 0 else 0

    # =========================================================================
    # STEP 4: Precision/Recall for FAIL (Critical per Eugene Yan!)
    # =========================================================================
    # Fail Recall is the KEY metric: catching failures is critical
    # Missing a defect (FN) is worse than a false alarm (FP)
    precision_fail = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_fail = tn / (tn + fp) if (tn + fp) > 0 else 0  # THE KEY METRIC

    # =========================================================================
    # STEP 5: F1 Scores
    # =========================================================================
    # Harmonic mean of precision and recall
    f1_pass = (
        2 * precision_pass * recall_pass / (precision_pass + recall_pass)
        if (precision_pass + recall_pass) > 0
        else 0
    )
    f1_fail = (
        2 * precision_fail * recall_fail / (precision_fail + recall_fail)
        if (precision_fail + recall_fail) > 0
        else 0
    )

    # =========================================================================
    # STEP 6: Cohen's Kappa
    # =========================================================================
    # Kappa measures agreement beyond what we'd expect by chance.
    # This is the MOST IMPORTANT metric for evaluator quality.
    #
    # Formula: κ = (po - pe) / (1 - pe)
    # - po = observed agreement (accuracy)
    # - pe = expected agreement by chance
    #
    # See docs/glossary.md#cohens-kappa
    # See examples/kappa_calculator.py for interactive exploration

    p_observed = accuracy

    # Expected agreement by chance
    # Based on marginal probabilities of each class
    p_pred_pass = (tp + fp) / total if total > 0 else 0
    p_true_pass = (tp + fn) / total if total > 0 else 0
    p_pred_fail = (tn + fn) / total if total > 0 else 0
    p_true_fail = (tn + fp) / total if total > 0 else 0

    p_expected = (p_pred_pass * p_true_pass) + (p_pred_fail * p_true_fail)

    # Kappa formula
    kappa = (p_observed - p_expected) / (1 - p_expected) if p_expected < 1 else 0

    # =========================================================================
    # STEP 7: Confidence Interval for Accuracy
    # =========================================================================
    # With small samples, metrics can be misleading.
    # The 95% CI shows the range where the true accuracy likely falls.
    # See docs/glossary.md#confidence-interval
    ci_lower, ci_upper = confidence_interval(accuracy, total)

    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "accuracy_ci": (ci_lower, ci_upper),
        "precision_pass": precision_pass,
        "recall_pass": recall_pass,
        "f1_pass": f1_pass,
        "precision_fail": precision_fail,
        "recall_fail": recall_fail,  # <-- Key metric!
        "f1_fail": f1_fail,
        "kappa": kappa,  # <-- Most important!
    }


def confidence_interval(
    accuracy: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Calculate confidence interval for accuracy.

    Uses the normal approximation to the binomial distribution.

    Args:
        accuracy: Observed accuracy (0 to 1)
        n: Number of samples
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (lower, upper) bounds

    Example:
        acc = 0.80
        n = 50
        lower, upper = confidence_interval(acc, n)
        print(f"Accuracy: {acc:.1%} (CI: {lower:.1%} - {upper:.1%})")
        # Output: Accuracy: 80.0% (CI: 68.9% - 91.1%)
    """
    if n == 0:
        return 0.0, 0.0

    # Z-score for confidence level
    z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645

    # Standard error
    se = math.sqrt(accuracy * (1 - accuracy) / n)

    # Confidence interval
    lower = max(0.0, accuracy - z * se)
    upper = min(1.0, accuracy + z * se)

    return lower, upper


def interpret_kappa(kappa: float) -> str:
    """
    Interpret Cohen's Kappa score per Eugene Yan's guidelines.

    Interpretation scale:
        < 0.0: Worse than chance (something is wrong)
        0.0-0.2: Slight agreement
        0.2-0.4: Fair agreement
        0.4-0.6: Substantial agreement (TARGET RANGE)
        0.6-0.8: Excellent agreement
        0.8-1.0: Near-perfect agreement (check for overfitting)

    Note:
        Human inter-rater reliability is often only 0.2-0.3!
        Getting 0.5 with an LLM evaluator is actually impressive.

    Args:
        kappa: Cohen's Kappa value

    Returns:
        Human-readable interpretation string

    Example:
        print(interpret_kappa(0.55))
        # Output: "Substantial agreement (TARGET)"
    """
    if kappa < 0:
        return "Worse than chance"
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
