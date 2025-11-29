# Metrics Explained

**Time:** 15 minutes

## Why Not Just Use Accuracy?

Consider this scenario:

```
Dataset: 100 samples
- 90 are "pass"
- 10 are "fail"

Evaluator: Always predicts "pass"

Accuracy: 90/100 = 90%
```

Is this evaluator good? **No!** It's useless - it just guesses the majority class.

This is why we need **Cohen's Kappa**.

## Cohen's Kappa

Measures agreement **beyond chance**.

### The Formula

```
κ = (po - pe) / (1 - pe)

Where:
- po = observed agreement (how often we matched)
- pe = expected agreement if both guessed randomly
```

### Intuition

- If `po = pe`: Kappa = 0 (no better than random)
- If `po = 1`: Kappa = 1 (perfect agreement)
- If `po < pe`: Kappa < 0 (worse than random!)

### Interpretation

| Kappa | Level | What It Means |
|-------|-------|---------------|
| < 0 | Worse than chance | Something is wrong |
| 0.0-0.2 | Slight | Not useful |
| 0.2-0.4 | Fair | Needs work |
| **0.4-0.6** | **Substantial** | **Target range** |
| 0.6-0.8 | Excellent | Very good |
| 0.8-1.0 | Near-perfect | Check for overfitting |

### Example Calculation

```python
# Run this to see the math:
uv run examples/kappa_calculator.py
```

Output:
```
Scenario: Always predicts pass (imbalanced)
  Predictions:   ['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']
  Ground Truth:  ['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'fail', 'fail']

  Results:
    Accuracy:    75.0%
    Kappa:       0.000 (Slight agreement)  ← Reveals the problem!
```

75% accuracy sounds okay, but Kappa = 0 shows it's no better than guessing.

## Confusion Matrix

A 2x2 table of outcomes:

```
                    Actual
                Pass      Fail
Predicted  ┌─────────┬─────────┐
   Pass    │   TP    │   FP    │
           │ (hit)   │ (false  │
           │         │  alarm) │
           ├─────────┼─────────┤
   Fail    │   FN    │   TN    │
           │ (miss)  │ (correct│
           │         │  reject)│
           └─────────┴─────────┘

TP = True Positive (correctly predicted pass)
TN = True Negative (correctly predicted fail)
FP = False Positive (predicted pass, was fail)
FN = False Negative (predicted fail, was pass)
```

## Precision vs Recall

### Precision: "When I say X, am I right?"

```
Pass Precision = TP / (TP + FP)
"Of all my 'pass' predictions, how many were correct?"

Fail Precision = TN / (TN + FN)
"Of all my 'fail' predictions, how many were correct?"
```

### Recall: "Did I catch all the X's?"

```
Pass Recall = TP / (TP + FN)
"Of all actual passes, how many did I find?"

Fail Recall = TN / (TN + FP)
"Of all actual failures, how many did I catch?"
```

## Why Prioritize Fail Recall?

Eugene Yan recommends prioritizing **fail recall** because:

1. **Missing defects is costly** - A bug in production is worse than a false alarm
2. **False negatives compound** - Missed failures become user complaints
3. **False positives are cheap** - Human can review and dismiss

Target: **Fail Recall > 80%**

```
Good Evaluator:
  Fail Recall: 85% (catches most failures)
  Fail Precision: 70% (some false alarms, but acceptable)

Bad Evaluator:
  Fail Recall: 50% (misses half the failures!)
  Fail Precision: 95% (rarely wrong, but rarely flags anything)
```

## F1 Score

Harmonic mean of precision and recall:

```
F1 = 2 * (precision * recall) / (precision + recall)
```

Useful when you need to balance both metrics.

## Confidence Interval

With small samples, metrics can be misleading. The 95% confidence interval shows uncertainty:

```
CI = accuracy ± 1.96 * sqrt(accuracy * (1 - accuracy) / n)

Example:
  Accuracy: 80%
  Samples: 50
  CI: 80% ± 11.1% = [68.9%, 91.1%]
```

More samples = tighter CI = more confidence.

## Metrics in This Project

When you run `verify_evaluator.py`, you get:

```
Metrics:
  Accuracy:        80.0% (CI: 74.5% - 85.5%)
  Cohen's Kappa:   0.558 (Substantial agreement)

  Pass Precision:  85.3%
  Pass Recall:     89.1%
  Pass F1:         87.2%

  Fail Precision:  68.4%
  Fail Recall:     69.0%  ← Watch this!
  Fail F1:         68.7%
```

## Key Takeaways

1. **Don't trust accuracy alone** - Misleading with imbalanced data
2. **Use Cohen's Kappa** - Measures agreement beyond chance
3. **Target Kappa 0.4-0.6** - Substantial agreement
4. **Prioritize fail recall** - Catching failures is critical
5. **Check confidence intervals** - Small samples have high uncertainty

## Next Steps

Now let's put it all together:

→ [06_full_pipeline.md](06_full_pipeline.md)

Or run a real evaluation:
```bash
uv run verify_evaluator.py --sample 50
```
