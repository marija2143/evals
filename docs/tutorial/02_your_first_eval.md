# Your First Eval

**Time:** 15 minutes

## Goal

Run a minimal evaluation: 1 sample, 1 evaluator, 1 metric.

## Prerequisites

```bash
# Install dependencies
uv sync

# Set up API key
export OPENAI_API_KEY=your-key-here
# Or add to .env file
```

## Run the Example

```bash
uv run examples/minimal_eval.py
```

Expected output:
```
==================================================
MINIMAL EVALUATION EXAMPLE
==================================================

Question: What is 2 + 2?
Response: The answer is 4.
Expected: pass

Predicted: pass
Match: Yes
```

## Understanding the Code

Open `examples/minimal_eval.py` and follow along:

### Step 1: Define the Sample

```python
SAMPLE = {
    "question": "What is 2 + 2?",
    "response": "The answer is 4.",
    "expected": "pass",  # Ground truth
}
```

This is our **ground truth** - we know the response is correct.

### Step 2: Define the Evaluation Tool

```python
EVAL_TOOL = {
    "type": "function",
    "name": "submit_evaluation",
    "strict": True,  # Enforces schema
    "parameters": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail"],  # Only these allowed
            }
        },
        "required": ["verdict"],
        "additionalProperties": False,
    },
}
```

Key points:
- `strict: True` ensures the model follows our schema
- `enum: ["pass", "fail"]` limits output to exactly these values
- `additionalProperties: False` prevents extra fields

### Step 3: Call the Evaluator

```python
result = client.responses.create(
    model="gpt-5-mini-2025-08-07",
    input=[{"role": "user", "content": prompt}],
    tools=[EVAL_TOOL],
    tool_choice={"type": "function", "name": "submit_evaluation"},
)
```

We're using:
- **Responses API** (not Chat Completions) for GPT-5.x
- **Function calling** with `tool_choice` to force the function

### Step 4: Extract the Result

```python
for item in result.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        verdict = args["verdict"]  # Guaranteed "pass" or "fail"
```

Because of `strict: True`, we're guaranteed valid JSON with exactly "pass" or "fail".

## Try It Yourself

1. **Modify the sample** - Change the response to be incorrect:
   ```python
   SAMPLE = {
       "question": "What is 2 + 2?",
       "response": "The answer is 5.",  # Wrong!
       "expected": "fail",
   }
   ```

2. **Run again** - Does the evaluator catch the error?

3. **Try edge cases**:
   ```python
   # Partial answer
   "response": "4"  # No explanation - pass or fail?

   # Verbose but wrong
   "response": "Let me think... 2 + 2 = 5 because..."
   ```

## What You Learned

- Evaluations compare model predictions to ground truth
- Function calling ensures reliable output
- `strict: True` + `enum` = guaranteed valid values
- One sample is enough to understand the pattern

## Next Steps

Now let's understand WHY function calling matters:

→ [03_function_calling.md](03_function_calling.md)

Or run the comparison yourself:
```bash
uv run examples/text_vs_function.py
```
