# Common Mistakes

Pitfalls to avoid when building LLM evaluations.

## 1. Text Parsing Instead of Function Calling

### The Mistake

```python
result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Reply with 'pass' or 'fail'"}],
)
text = result.choices[0].message.content

# Fragile parsing
verdict = "pass" if "pass" in text.lower() else "fail"
```

### Why It Fails

The model might return:
- `"Pass"` - Works, but fragile
- `"PASS"` - Works with `.lower()`
- `"I would pass this response"` - Contains "pass" but not the answer
- `"The bypass fails"` - Contains "pass" (false positive!)
- `"compassionate"` - Contains "pass" (false positive!)

### The Fix

Use function calling with strict schema:

```python
TOOL = {
    "type": "function",
    "name": "submit_evaluation",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "fail"]}
        },
        "required": ["verdict"],
        "additionalProperties": False,
    },
}

result = client.responses.create(
    model="gpt-5-mini-2025-08-07",
    input=[...],
    tools=[TOOL],
    tool_choice={"type": "function", "name": "submit_evaluation"},
)
```

---

## 2. Forgetting `additionalProperties: false`

### The Mistake

```python
TOOL = {
    "strict": True,
    "parameters": {
        "properties": {
            "verdict": {"enum": ["pass", "fail"]}
        },
        "required": ["verdict"],
        # Missing: "additionalProperties": False
    },
}
```

### Why It Fails

Without `additionalProperties: false`, the model can return extra fields:

```json
{
    "verdict": "pass",
    "confidence": 0.9,     // Unexpected!
    "reasoning": "..."     // Unexpected!
}
```

This can break your parsing or cause silent errors.

### The Fix

Always include `additionalProperties: false` for strict mode:

```python
"parameters": {
    "type": "object",
    "properties": {...},
    "required": [...],
    "additionalProperties": False,  # Always include this
}
```

---

## 3. Using Accuracy Instead of Kappa

### The Mistake

```python
correct = sum(pred == truth for pred, truth in zip(predictions, ground_truth))
accuracy = correct / len(predictions)
print(f"Accuracy: {accuracy:.1%}")  # "Accuracy: 90%!"
```

### Why It Fails

With imbalanced data:
- 90% of samples are "pass"
- Always predicting "pass" = 90% accuracy
- But the evaluator is useless!

### The Fix

Use Cohen's Kappa:

```python
def calculate_kappa(predictions, ground_truth):
    n = len(predictions)

    # Observed agreement
    po = sum(p == g for p, g in zip(predictions, ground_truth)) / n

    # Expected agreement by chance
    p_pred_pass = predictions.count("pass") / n
    p_true_pass = ground_truth.count("pass") / n
    pe = (p_pred_pass * p_true_pass) + ((1 - p_pred_pass) * (1 - p_true_pass))

    # Kappa
    return (po - pe) / (1 - pe) if pe != 1 else 0
```

Target: Kappa 0.4-0.6 (substantial agreement)

---

## 4. Ignoring Position Bias

### The Mistake

```python
def compare(response_a, response_b):
    result = llm.compare(f"A: {response_a}\nB: {response_b}\nWhich is better?")
    return result  # Trust this blindly
```

### Why It Fails

Models often prefer responses based on position (first or last), not quality.

### The Fix

Run twice with swapped order:

```python
def compare_with_bias_detection(response_a, response_b):
    result1 = compare(response_a, response_b)  # A first
    result2 = compare(response_b, response_a)  # B first

    if result1 == result2:
        return result1  # Consistent = genuine preference
    else:
        return "tie"    # Inconsistent = position bias
```

---

## 5. Using Wrong API Parameters

### The Mistake

```python
# Using Chat Completions params with Responses API
client.responses.create(
    model="gpt-5-mini-2025-08-07",
    messages=[...],        # ❌ Wrong - should be "input"
    max_tokens=100,        # ❌ Wrong - should be "max_completion_tokens"
)
```

### Why It Fails

Different APIs have different parameter names. Using the wrong ones causes errors or silent failures.

### The Fix

| API | Messages | Token Limit |
|-----|----------|-------------|
| Chat Completions | `messages=[...]` | `max_tokens=100` |
| Responses | `input=[...]` | `max_completion_tokens=100` |

---

## 6. Not Handling Rate Limits

### The Mistake

```python
for sample in samples:
    result = client.responses.create(...)  # No error handling
```

### Why It Fails

APIs have rate limits. Without handling, your script crashes partway through.

### The Fix

Exponential backoff:

```python
import time
from openai import RateLimitError

def call_with_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            wait = 3 ** attempt
            print(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
    raise Exception("Max retries exceeded")
```

---

## 7. Small Sample Sizes Without Confidence Intervals

### The Mistake

```python
# 10 samples
accuracy = 8 / 10  # 80%
print("Our evaluator has 80% accuracy!")
```

### Why It Fails

With small samples, 80% could actually be anywhere from 55% to 95%.

### The Fix

Always report confidence intervals:

```python
import math

def confidence_interval(accuracy, n, confidence=0.95):
    z = 1.96  # For 95% CI
    se = math.sqrt(accuracy * (1 - accuracy) / n)
    return (accuracy - z * se, accuracy + z * se)

acc = 0.80
n = 10
ci = confidence_interval(acc, n)
print(f"Accuracy: {acc:.1%} (CI: {ci[0]:.1%} - {ci[1]:.1%})")
# Output: Accuracy: 80.0% (CI: 55.2% - 100.0%)
```

Eugene Yan recommends 200+ samples for meaningful signal.

---

## 8. Building a "God Evaluator"

### The Mistake

```python
def evaluate(response):
    prompt = """
    Evaluate this response on:
    - Accuracy
    - Helpfulness
    - Safety
    - Clarity
    - Conciseness
    - Tone
    Return scores for each.
    """
```

### Why It Fails

- Too many dimensions = confusion
- Hard to iterate on specific issues
- Unclear which dimension failed
- Kappa is hard to interpret

### The Fix

One evaluator per dimension:

```python
def evaluate_accuracy(response):
    """Is the response factually correct?"""
    ...

def evaluate_helpfulness(response):
    """Does the response address the user's need?"""
    ...

def evaluate_safety(response):
    """Is the response free of harmful content?"""
    ...
```

Each evaluator is simple, focused, and independently measurable.

---

## 9. Not Prioritizing Fail Recall

### The Mistake

```python
# Optimizing for overall accuracy
if accuracy > 0.9:
    print("Great evaluator!")
```

### Why It Fails

Missing failures (false negatives) is usually more costly than false alarms (false positives).

### The Fix

Prioritize fail recall:

```python
fail_recall = tn / (tn + fn)  # Of all failures, how many did we catch?

if fail_recall > 0.8:
    print("Good at catching failures!")
elif fail_recall < 0.6:
    print("Warning: Missing too many failures!")
```

---

## 10. Hardcoding Ground Truth Labels

### The Mistake

```python
# Manually labeling in code
SAMPLES = [
    {"question": "...", "response": "...", "label": "pass"},  # "This looks right"
    {"question": "...", "response": "...", "label": "fail"},  # "I think this is wrong"
]
```

### Why It Fails

- Subjective and inconsistent
- Not scalable
- No audit trail
- Can't update systematically

### The Fix

Use a stronger model with function calling for ground truth:

```bash
# Generate with weaker model (GPT-3.5)
uv run scripts/generate_hard_questions.py

# Label with stronger model (GPT-5.1)
uv run scripts/label_responses.py
```

This creates reproducible, consistent labels with clear methodology.

---

## Quick Checklist

Before deploying an evaluator:

- [ ] Using function calling with strict schema?
- [ ] `additionalProperties: false` included?
- [ ] Using Cohen's Kappa, not just accuracy?
- [ ] Position bias detection for comparisons?
- [ ] Correct API parameters for your model?
- [ ] Rate limit handling with exponential backoff?
- [ ] 200+ samples with confidence intervals?
- [ ] One evaluator per dimension?
- [ ] Fail recall > 80%?
- [ ] Ground truth from stronger model, not manual?
