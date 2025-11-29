# Function Calling vs Text Parsing

**Time:** 10 minutes

## The Problem with Text Parsing

Imagine you ask an LLM to evaluate a response:

```
Prompt: "Is this answer correct? Reply with 'pass' or 'fail'."
```

What you might get back:

```
"Pass"                              # Works
"PASS"                              # Case issue
"pass"                              # Works
"I would pass this response"        # Embedded in sentence
"The response passes the test"      # Verb form
"Verdict: Pass."                    # With punctuation
"It's a passing grade"              # Adjective form
"I think it passes, but..."         # Qualified
```

### Naive Parsing Breaks

```python
# Fragile approach
verdict = "pass" if "pass" in response.lower() else "fail"
```

This fails on:
- `"The bypass fails"` → False positive (contains "pass")
- `"compassionate response"` → False positive
- `"I can't pass judgment"` → False positive

## Function Calling Solves This

With function calling, you define a **schema** that the model must follow:

```python
{
    "type": "function",
    "name": "submit_evaluation",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail"]
            }
        },
        "required": ["verdict"],
        "additionalProperties": False
    }
}
```

Now the model MUST return:
```json
{"verdict": "pass"}
// or
{"verdict": "fail"}
```

No parsing needed. No edge cases. Guaranteed.

## See It in Action

```bash
uv run examples/text_vs_function.py
```

Output:
```
PROBLEM: Text parsing is fragile
----------------------------------------
  'I would pass this response'... -> pass
  'PASS'... -> pass
  ...

  All above parse correctly, but what about:
    'The bypass fails' ->  pass   # WRONG!
    'compassionate response' ->  pass   # WRONG!
  False positives from 'pass' substring!

COMPARISON: Actual API calls
----------------------------------------

1. Text Parsing (Chat Completions API):
   Raw output: 'Pass'
   Parsed as: 'pass'

2. Function Calling (Responses API):
   Raw output: {"verdict": "pass"}
   Parsed as: 'pass' (guaranteed valid)
```

## Strict Mode Requirements

For reliable function calling, you need ALL of these:

```python
{
    "strict": True,                    # 1. Enable strict mode
    "parameters": {
        "properties": {
            "verdict": {
                "enum": ["pass", "fail"]  # 2. Limit to specific values
            }
        },
        "additionalProperties": False   # 3. No extra fields
    }
}
```

Missing any of these can lead to unexpected output.

## API Differences

| Aspect | Chat Completions | Responses API |
|--------|------------------|---------------|
| Parameter | `messages=[...]` | `input=[...]` |
| Token limit | `max_tokens` | `max_completion_tokens` |
| Models | GPT-4o-mini, 3.5 | GPT-5.x |
| Best for | Text generation | Function calling |

## When to Use What

| Use Case | Approach |
|----------|----------|
| Pass/fail evaluation | Function calling (strict) |
| Yes/no classification | Function calling (strict) |
| Category selection | Function calling (enum) |
| Free-form feedback | Text (no parsing needed) |
| Scores with reasoning | Function calling (structured) |

## Key Takeaways

1. **Never parse free-form text** for structured output
2. **Always use strict mode** with enums for classification
3. **Responses API** is better for function calling with GPT-5.x
4. **Test edge cases** - substring matches cause false positives

## Next Steps

Now let's understand position bias:

→ [04_position_bias.md](04_position_bias.md)

Or run the demo:
```bash
uv run examples/position_bias_demo.py
```
