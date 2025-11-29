#!/usr/bin/env python3
"""
Understanding LLM Evaluations - Interactive Notebook
=====================================================
Run with: uv run marimo edit notebooks/01_understanding_evals.py

This notebook teaches the fundamentals of LLM evaluation through
interactive examples. No API calls needed for the core concepts.
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Understanding LLM Evaluations

        This interactive notebook teaches you the core concepts of evaluating LLM outputs.

        **What you'll learn:**
        1. Why accuracy is misleading
        2. How Cohen's Kappa works
        3. Confusion matrices explained
        4. Why fail recall matters

        No API calls needed - pure Python for understanding the math.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        ## The Problem with Accuracy

        Imagine you have a dataset where **90% of responses are correct** (pass).

        An evaluator that **always predicts "pass"** gets 90% accuracy!

        But is it useful? Let's see...
        """
    )
    return


@app.cell
def _():
    # Simulate an imbalanced dataset
    ground_truth = ["pass"] * 90 + ["fail"] * 10  # 90% pass, 10% fail

    # "Always pass" evaluator
    always_pass_predictions = ["pass"] * 100

    # Calculate accuracy
    correct = sum(p == g for p, g in zip(always_pass_predictions, ground_truth))
    accuracy = correct / len(ground_truth)

    print(f"Ground truth: {ground_truth.count('pass')} pass, {ground_truth.count('fail')} fail")
    print(f"Predictions:  Always 'pass'")
    print(f"Accuracy:     {accuracy:.1%}")
    print()
    print("90% accuracy sounds great, but this evaluator is useless!")
    print("It never identifies failures - the whole point of evaluation!")
    return accuracy, always_pass_predictions, correct, ground_truth


@app.cell
def _(mo):
    mo.md(
        """
        ## Cohen's Kappa: The Solution

        Cohen's Kappa measures agreement **beyond what we'd expect by chance**.

        **Formula:**
        ```
        κ = (observed agreement - expected agreement) / (1 - expected agreement)
        ```

        - **Kappa = 0**: No better than random guessing
        - **Kappa = 1**: Perfect agreement
        - **Kappa < 0**: Worse than random!

        Let's calculate it step by step...
        """
    )
    return


@app.cell
def _():
    def calculate_kappa_detailed(predictions: list, ground_truth: list) -> dict:
        """Calculate Cohen's Kappa with detailed breakdown."""
        n = len(predictions)

        # Confusion matrix
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == "pass" and g == "pass")
        tn = sum(1 for p, g in zip(predictions, ground_truth) if p == "fail" and g == "fail")
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == "pass" and g == "fail")
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p == "fail" and g == "pass")

        # Observed agreement
        po = (tp + tn) / n

        # Marginal probabilities
        p_pred_pass = (tp + fp) / n
        p_true_pass = (tp + fn) / n
        p_pred_fail = (tn + fn) / n
        p_true_fail = (tn + fp) / n

        # Expected agreement by chance
        pe = (p_pred_pass * p_true_pass) + (p_pred_fail * p_true_fail)

        # Kappa
        kappa = (po - pe) / (1 - pe) if pe != 1 else 0

        return {
            "accuracy": po,
            "observed_agreement": po,
            "expected_agreement": pe,
            "kappa": kappa,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }

    return (calculate_kappa_detailed,)


@app.cell
def _(calculate_kappa_detailed):
    # Same imbalanced scenario
    ground_truth_imb = ["pass"] * 90 + ["fail"] * 10
    always_pass = ["pass"] * 100

    result = calculate_kappa_detailed(always_pass, ground_truth_imb)

    print("Scenario: Always predict 'pass' with 90% pass rate")
    print("=" * 50)
    print(f"Observed agreement (po): {result['observed_agreement']:.3f}")
    print(f"Expected by chance (pe): {result['expected_agreement']:.3f}")
    print(f"Cohen's Kappa:          {result['kappa']:.3f}")
    print()
    print("Interpretation:")
    if result["kappa"] == 0:
        print("  Kappa = 0 means NO BETTER THAN CHANCE!")
    print("  The evaluator is useless despite 90% accuracy.")
    return always_pass, ground_truth_imb, result


@app.cell
def _(mo):
    mo.md(
        """
        ## Interactive: Try Different Scenarios

        Use the sliders below to explore how Kappa changes with different prediction patterns.
        """
    )
    return


@app.cell
def _(mo):
    # Create sliders for interactive exploration
    tp_slider = mo.ui.slider(0, 50, value=40, label="True Positives (correctly identified passes)")
    tn_slider = mo.ui.slider(0, 50, value=8, label="True Negatives (correctly identified failures)")
    fp_slider = mo.ui.slider(0, 50, value=2, label="False Positives (false alarms)")
    fn_slider = mo.ui.slider(0, 50, value=0, label="False Negatives (missed failures)")

    mo.vstack([tp_slider, tn_slider, fp_slider, fn_slider])
    return fn_slider, fp_slider, tn_slider, tp_slider


@app.cell
def _(fn_slider, fp_slider, mo, tn_slider, tp_slider):
    tp = tp_slider.value
    tn = tn_slider.value
    fp = fp_slider.value
    fn = fn_slider.value
    n = tp + tn + fp + fn

    if n == 0:
        mo.md("**Add some samples using the sliders above!**")
    else:
        # Calculate metrics
        accuracy = (tp + tn) / n
        po = accuracy  # Observed agreement

        # Expected agreement
        p_pred_pass = (tp + fp) / n if n > 0 else 0
        p_true_pass = (tp + fn) / n if n > 0 else 0
        p_pred_fail = (tn + fn) / n if n > 0 else 0
        p_true_fail = (tn + fp) / n if n > 0 else 0
        pe = (p_pred_pass * p_true_pass) + (p_pred_fail * p_true_fail)

        kappa = (po - pe) / (1 - pe) if pe != 1 else 0

        # Fail recall
        fail_recall = tn / (tn + fn) if (tn + fn) > 0 else 0

        # Interpretation
        if kappa < 0:
            interp = "Worse than chance"
        elif kappa < 0.2:
            interp = "Slight agreement"
        elif kappa < 0.4:
            interp = "Fair agreement"
        elif kappa < 0.6:
            interp = "Substantial agreement (TARGET)"
        elif kappa < 0.8:
            interp = "Excellent agreement"
        else:
            interp = "Near-perfect agreement"

        mo.md(
            f"""
            ## Results

            **Confusion Matrix:**
            ```
                            Actual
                        Pass    Fail
            Predicted
                Pass    {tp:3d}     {fp:3d}
                Fail    {fn:3d}     {tn:3d}
            ```

            **Metrics:**
            | Metric | Value | Notes |
            |--------|-------|-------|
            | Accuracy | {accuracy:.1%} | Simple % correct |
            | Cohen's Kappa | {kappa:.3f} | {interp} |
            | Fail Recall | {fail_recall:.1%} | % of failures caught (target: >80%) |
            | Samples | {n} | Total |

            **Why this matters:**
            - If Kappa ≈ 0 but accuracy is high → evaluator is just guessing the majority class
            - Target: Kappa 0.4-0.6 (substantial agreement)
            - Prioritize fail recall > 80% (catching failures is critical)
            """
        )
    return (
        accuracy,
        fail_recall,
        fn,
        fp,
        interp,
        kappa,
        n,
        p_pred_fail,
        p_pred_pass,
        p_true_fail,
        p_true_pass,
        pe,
        po,
        tn,
        tp,
    )


@app.cell
def _(mo):
    mo.md(
        """
        ## Key Takeaways

        1. **Accuracy is misleading** with imbalanced data (most real datasets are imbalanced)

        2. **Cohen's Kappa** accounts for chance agreement:
           - Kappa = 0: No better than guessing
           - Kappa 0.4-0.6: Target range (substantial agreement)
           - Kappa > 0.6: Excellent

        3. **Fail Recall matters more** than overall accuracy:
           - Missing defects (false negatives) is costly
           - False alarms (false positives) are cheap to review

        4. **Human baseline**: Inter-rater reliability is often only Kappa 0.2-0.3

        ---

        **Next:** Open `notebooks/02_position_bias.py` to learn about position bias
        """
    )
    return


if __name__ == "__main__":
    app.run()
