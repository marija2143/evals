# Why Evals Matter

**Time:** 10 minutes

## The Problem

You've built an LLM-powered feature. It seems to work. How do you know if it's actually good?

```
User: "Summarize this article"
LLM: "This article discusses various topics..."

Is this a good summary? How would you know?
```

## The Naive Approach (Don't Do This)

```python
# "It looks good to me"
response = llm.generate(prompt)
if response:
    print("Ship it!")
```

Problems:
- **No measurement** → Can't improve what you can't measure
- **Subjective** → Your "good" isn't everyone's "good"
- **Not scalable** → Can't manually check 1000 responses
- **Regression blind** → Model updates might break things

## Eugene Yan's Methodology

[Eugene Yan](https://eugeneyan.com/writing/product-evals/) proposes a practical approach:

### 1. Binary Labels, Not Scales

```
❌ Rate 1-5
✅ Pass or Fail
```

Why? Numeric scales are subjective. Is a 3 good or bad? Binary forces clear decisions.

### 2. Generate Organic Failures

```
❌ Manually write "bad" examples
✅ Use a weaker model to generate failures naturally
```

GPT-3.5 answering hard questions creates realistic failure cases.

### 3. One Evaluator Per Dimension

```
❌ "Is this response good?" (too vague)
✅ "Is this response accurate?" (specific)
✅ "Is this response helpful?" (specific)
```

Each evaluator checks ONE thing.

### 4. Measure Agreement, Not Accuracy

```
❌ "90% accuracy!" (misleading with imbalanced data)
✅ "Cohen's Kappa 0.5" (accounts for chance)
```

If 90% of samples pass, always predicting "pass" = 90% accuracy. Useless!

### 5. Prioritize Fail Recall

```
❌ Optimize for overall accuracy
✅ Ensure we catch failures (false negatives are costly)
```

Missing a defect in production is worse than a false alarm.

## The Three Steps

```
STEP 1: Label Data
        Create samples with ground truth (pass/fail)

STEP 2: Build Evaluators
        LLM judges that predict pass/fail

STEP 3: Measure Agreement
        Compare evaluator predictions to ground truth
        Calculate Cohen's Kappa (target: 0.4-0.6)
```

## Key Metrics

| Metric | What It Measures | Target |
|--------|------------------|--------|
| Cohen's Kappa | Agreement beyond chance | 0.4-0.6 |
| Fail Recall | % of failures caught | >80% |
| Accuracy | Simple % correct | N/A (misleading) |

## Human Baseline

Here's the surprising part: human inter-rater reliability is often only **Kappa 0.2-0.3**.

Getting Kappa 0.5 with an LLM evaluator is actually impressive!

## Next Steps

Now that you understand WHY evals matter, let's build one:

→ [02_your_first_eval.md](02_your_first_eval.md)
