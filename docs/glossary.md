# Glossary

Essential terminology for understanding LLM evaluations. Based on [Eugene Yan's methodology](https://eugeneyan.com/writing/product-evals/).

## Core Concepts

### Cohen's Kappa

**Definition:** A statistical measure of agreement between two raters (your evaluator vs ground truth) that accounts for chance agreement.

**Formula:** `κ = (po - pe) / (1 - pe)`
- `po` = observed agreement (% of samples where both agree)
- `pe` = expected agreement by chance

**Interpretation:**
| Kappa | Agreement Level | Notes |
|-------|-----------------|-------|
| < 0 | Worse than chance | Systematic disagreement |
| 0.0 - 0.2 | Slight | Not useful |
| 0.2 - 0.4 | Fair | Needs improvement |
| 0.4 - 0.6 | Substantial | **Target range** |
| 0.6 - 0.8 | Excellent | Very good |
| 0.8 - 1.0 | Near-perfect | Suspicious (check for overfitting) |

**Why it matters:** Accuracy is misleading with imbalanced data. If 90% of samples are "pass", always predicting "pass" gives 90% accuracy but Kappa = 0.

**Example:** See `examples/kappa_calculator.py`

---

### Position Bias

**Definition:** The tendency of LLMs to prefer responses based on their position in the prompt (first vs last) rather than their actual quality.

**Detection method:**
1. Compare A vs B (A shown first)
2. Compare B vs A (order swapped)
3. If answers differ → position bias detected → return "tie"

**Why it matters:** Without detection, A/B tests and comparisons become unreliable.

**Example:** See `examples/position_bias_demo.py`

---

### Ground Truth

**Definition:** The "correct" labels against which we measure our evaluator's performance. Typically created by:
- Human experts (gold standard)
- Stronger model (e.g., GPT-5.1 labeling data for GPT-5-mini to evaluate)

**In this project:** Ground truth labels are in `data/questions_version_2.csv`, created using GPT-5.1 function calling.

---

### Fail Recall (Sensitivity)

**Definition:** The percentage of actual failures that the evaluator correctly identifies.

**Formula:** `Fail Recall = TN / (TN + FN)`
- `TN` = True Negatives (correctly identified failures)
- `FN` = False Negatives (failures missed, predicted as pass)

**Why it matters:** In production, missing defects (false negatives) is usually more costly than false alarms (false positives). Eugene Yan recommends prioritizing fail recall.

**Target:** > 80%

---

### Strict Schema

**Definition:** A function calling configuration that guarantees the model returns exactly the values you specify.

**Required components:**
```python
{
    "strict": True,
    "parameters": {
        "properties": {
            "verdict": {
                "enum": ["pass", "fail"]  # Only these values allowed
            }
        },
        "additionalProperties": False  # No extra fields
    }
}
```

**Why it matters:** Without strict schema, the model might return "Pass", "PASS", "I think it passes", etc. - requiring fragile text parsing.

**Example:** See `examples/text_vs_function.py`

---

## Metrics

### Confusion Matrix

A 2x2 table showing prediction outcomes:

```
                  Actual
              Pass    Fail
Predicted  ┌────────┬────────┐
   Pass    │   TP   │   FP   │
           ├────────┼────────┤
   Fail    │   FN   │   TN   │
           └────────┴────────┘
```

- **TP (True Positive):** Correctly predicted pass
- **TN (True Negative):** Correctly predicted fail
- **FP (False Positive):** Predicted pass, actually fail (false alarm)
- **FN (False Negative):** Predicted fail, actually pass (missed detection)

---

### Precision

**Definition:** Of all samples predicted as a class, what % were correct?

- **Pass Precision:** `TP / (TP + FP)` — When we say "pass", how often are we right?
- **Fail Precision:** `TN / (TN + FN)` — When we say "fail", how often are we right?

---

### Recall (Sensitivity)

**Definition:** Of all actual samples of a class, what % did we catch?

- **Pass Recall:** `TP / (TP + FN)` — Of all actual passes, how many did we identify?
- **Fail Recall:** `TN / (TN + FP)` — Of all actual failures, how many did we catch?

---

### F1 Score

**Definition:** Harmonic mean of precision and recall. Balances both metrics.

**Formula:** `F1 = 2 * (precision * recall) / (precision + recall)`

---

### Confidence Interval

**Definition:** Range within which the true metric likely falls (typically 95% confidence).

**Formula:** `CI = accuracy ± 1.96 * sqrt(accuracy * (1 - accuracy) / n)`

**Why it matters:** With small samples, metrics can be misleading. CI shows uncertainty.

---

## APIs

### Chat Completions API

OpenAI's original API for conversational models (GPT-4o-mini, GPT-3.5).

```python
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],    # "messages" parameter
    max_tokens=100,    # "max_tokens" parameter
)
```

---

### Responses API

OpenAI's newer API for GPT-5.x models. Better for function calling.

```python
client.responses.create(
    model="gpt-5.1-2025-11-13",
    input=[...],              # "input" not "messages"
    max_completion_tokens=100, # Different parameter name
    tools=[...],
    tool_choice={...},
)
```

---

### Function Calling

**Definition:** A mechanism where the model returns structured data by "calling" a predefined function schema.

**Benefits:**
- Guaranteed output format (no parsing)
- Type-safe values (enums, numbers, etc.)
- Works with strict mode for reliability

---

## Eugene Yan's Principles

From [Product Evals in Three Simple Steps](https://eugeneyan.com/writing/product-evals/):

1. **Binary labels:** Use pass/fail, not numeric scales (1-5)
2. **Organic failures:** Use weaker models to generate natural failure cases
3. **One evaluator per dimension:** Don't build a "God Evaluator"
4. **Target failure rate:** 50-100 failures in 200+ samples
5. **Prioritize fail recall:** Catching defects is critical
6. **Cohen's Kappa 0.4-0.6:** Substantial agreement is the target
7. **Human baseline:** Inter-rater reliability is often only 0.2-0.3

---

## Quick Reference

| Term | One-liner |
|------|-----------|
| Cohen's Kappa | Agreement beyond chance (target: 0.4-0.6) |
| Position Bias | Model prefers first/last option regardless of content |
| Ground Truth | Human-verified or stronger-model labels |
| Fail Recall | % of failures caught (target: >80%) |
| Strict Schema | `strict: true` + `enum` + `additionalProperties: false` |
| TP/TN/FP/FN | Confusion matrix quadrants |
| Precision | "When I say X, am I right?" |
| Recall | "Did I catch all the X's?" |
