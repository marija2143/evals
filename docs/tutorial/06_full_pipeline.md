# Full Pipeline

**Time:** 20 minutes

## Overview

Now you understand the concepts. Let's run the complete pipeline.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Generate   │────▶│   Label     │────▶│   Verify    │────▶│  Visualize  │
│  Questions  │     │  Responses  │     │  Evaluator  │     │   Results   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Step 1: Explore the Dataset

First, let's look at the existing dataset:

```bash
head -5 data/questions_version_2.csv
```

```csv
id,question,response,label,category
1,"Which city is the true geographical center of the EU?","The true geographical center...","fail","ambiguous_debatable"
2,"What is the most widely spoken language in the world?","Mandarin Chinese is the most...","pass","ambiguous_debatable"
...
```

Dataset stats:
- 200 samples
- 35.5% failure rate (71 fails)
- 6 categories

## Step 2: Run a Quick Evaluation

```bash
uv run verify_evaluator.py --sample 20
```

This evaluates 20 random samples with GPT-5-mini and shows:

```
Evaluating 20 samples with gpt-5-mini-2025-08-07...
[████████████████████] 100% 20/20

VERIFICATION RESULTS
====================

Confusion Matrix:
                 Actual
            Pass    Fail
Predicted
    Pass     12       2
    Fail      1       5

Metrics:
  Accuracy: 85.0% (CI: 70.2% - 99.8%)
  Cohen's Kappa: 0.667 (Excellent agreement)

  Fail Recall: 71.4%
  Fail Precision: 83.3%
```

## Step 3: Run Full Evaluation

```bash
uv run verify_evaluator.py
```

This runs all 200 samples. Takes a few minutes.

## Step 4: Compare Models

Try different models:

```bash
# Cheaper, faster model
uv run verify_evaluator.py --model gpt-5-nano-2025-08-07

# Compare to Cerebras (if configured)
uv run verify_evaluator.py --model llama-3.3-70b
```

## Step 5: Filter by Category

See which categories are hardest:

```bash
uv run verify_evaluator.py --category precise_calculations
uv run verify_evaluator.py --category temporal_changing
```

Expected pattern:
- `precise_calculations`: Lower accuracy (math is hard)
- `temporal_changing`: Lower accuracy (current events)
- `trick_misconceptions`: Higher accuracy (clear right/wrong)

## Step 6: Visualize Results

After running multiple evaluations:

```bash
uv run plot_runs.py
```

This generates:
- `docs/plots/model_comparison.png` - Bar chart of metrics
- `docs/plots/category_comparison.png` - Heatmap by category
- `docs/plots/confusion_matrices.png` - Visual confusion matrices

## Step 7: Understand the Demo Scripts

Compare the three implementations:

```bash
# Best practices (recommended)
uv run eval_demo_gpt5.py

# Shows text parsing issues
uv run eval_demo_gpt4.py

# Free tier option
uv run eval_demo_gemini.py
```

Key differences:

| Script | API | Method | Reliability |
|--------|-----|--------|-------------|
| `eval_demo_gpt5.py` | Responses | Function calling | High |
| `eval_demo_gpt4.py` | Chat Completions | Text parsing | Low |
| `eval_demo_gemini.py` | Gemini | Text parsing | Low |

## Step 8: Generate Your Own Data

If you want to create a new dataset:

```bash
# 1. Generate questions and weak answers
uv run scripts/generate_hard_questions.py

# 2. Label with ground truth
uv run scripts/label_responses.py
```

This creates questions designed for high failure rates using:
- GPT-5.1 to generate challenging questions
- GPT-3.5 to generate answers (weaker model = more failures)
- GPT-5.1 to label pass/fail (ground truth)

## Interactive Exploration

For deeper exploration, use the marimo notebooks:

```bash
# Understanding evals
uv run marimo edit notebooks/01_understanding_evals.py

# Position bias experiments
uv run marimo edit notebooks/02_position_bias.py

# Kappa visualizations
uv run marimo edit notebooks/03_kappa_intuition.py
```

## Checklist: Production Eval System

Use this checklist when building your own:

- [ ] **Binary labels** - Pass/fail, not scales
- [ ] **Ground truth** - Human-verified or stronger model labels
- [ ] **Function calling** - Strict schema with enums
- [ ] **Position bias detection** - Run comparisons twice
- [ ] **Cohen's Kappa** - Not just accuracy
- [ ] **Fail recall priority** - Catch defects
- [ ] **Per-category breakdown** - Find weak spots
- [ ] **Confidence intervals** - Know your uncertainty
- [ ] **Historical tracking** - JSONL logs for trends
- [ ] **Visualization** - Plots for communication

## What You've Learned

1. **Why evals matter** - Measure before you ship
2. **Function calling** - Reliable structured output
3. **Position bias** - Run twice, detect inconsistency
4. **Metrics** - Kappa > accuracy, fail recall > precision
5. **Full pipeline** - Generate → Label → Verify → Visualize

## Next Steps

- Read the API comparison: `docs/api_comparison.md`
- Check common mistakes: `docs/common_mistakes.md`
- Explore the glossary: `docs/glossary.md`
- Read Eugene Yan's original post: [eugeneyan.com/writing/product-evals/](https://eugeneyan.com/writing/product-evals/)

Congratulations! You now understand production-grade LLM evaluation.
