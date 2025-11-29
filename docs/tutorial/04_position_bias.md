# Position Bias

**Time:** 10 minutes

## What is Position Bias?

LLMs sometimes prefer responses based on their **position** in the prompt (first vs last) rather than their actual quality.

```
Prompt: "Which is better, A or B?"

If A is shown first, model might prefer A.
If B is shown first, model might prefer B.

This is position bias - the order affects the answer!
```

## Why It Matters

Without detecting position bias:
- **A/B tests become unreliable** - You might ship the worse option
- **Comparisons are inconsistent** - Results depend on random ordering
- **Evaluations are misleading** - High scores don't mean high quality

## Detection Method

Run the comparison **twice** with the order swapped:

```
Round 1: "Compare A vs B" (A first)
         → Winner: B

Round 2: "Compare B vs A" (B first)
         → Winner: B

Same answer? ✓ No position bias - B is genuinely better
```

```
Round 1: "Compare A vs B" (A first)
         → Winner: A

Round 2: "Compare B vs A" (B first)
         → Winner: B  ← Different!

Different answers? ✗ Position bias detected - return "tie"
```

## See It in Action

```bash
uv run examples/position_bias_demo.py
```

The demo compares two responses:
- **Response A:** "A variable stores data." (too brief)
- **Response B:** "A variable is a named container that..." (detailed)

Response B is clearly better. The demo checks if the model agrees regardless of order.

## Implementation Pattern

```python
def compare_with_bias_detection(response_a, response_b):
    # Round 1: Original order
    result_1 = compare(response_a, response_b, labels=("A", "B"))

    # Round 2: Swapped order
    result_2 = compare(response_b, response_a, labels=("B", "A"))

    # Check for consistency
    if result_1 == result_2:
        return result_1  # Genuine preference
    else:
        return "tie"     # Position bias detected
```

## In Production Code

See `eval_demo_gpt5.py` for the full implementation:

```python
def compare_responses(response_a: str, response_b: str) -> str:
    # First comparison: A first, B second
    result1 = _compare(response_a, response_b, "A", "B")

    # Second comparison: B first, A second
    result2 = _compare(response_b, response_a, "B", "A")

    if result1 == result2:
        return result1
    return "tie"  # Position bias
```

## Reducing Position Bias

Some techniques to minimize position bias:

1. **Always run twice** - Detect and handle with "tie"
2. **Randomize order** - But still run twice to detect
3. **Use function calling** - Structured output reduces bias
4. **Newer models** - GPT-5.x has less position bias than older models

## Measuring Position Bias Rate

Track how often you get "tie" due to position bias:

```python
results = [compare_with_bias_detection(a, b) for a, b in pairs]
bias_rate = results.count("tie") / len(results)
print(f"Position bias rate: {bias_rate:.1%}")
```

High bias rate (>20%) suggests:
- Responses are too similar to distinguish
- Model has strong position preference
- Prompt needs improvement

## Key Takeaways

1. **Position bias is real** - Order affects LLM preferences
2. **Always run comparisons twice** - Swap order, check consistency
3. **Return "tie" on disagreement** - Don't trust biased results
4. **Track bias rate** - High rates indicate problems

## Next Steps

Now let's understand the metrics:

→ [05_metrics_explained.md](05_metrics_explained.md)

Or explore metrics interactively:
```bash
uv run examples/kappa_calculator.py
```
