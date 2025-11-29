# API Comparison

Quick reference for the different APIs used in this project.

## OpenAI APIs

### Chat Completions vs Responses API

| Aspect | Chat Completions | Responses API |
|--------|------------------|---------------|
| **Models** | GPT-4o-mini, GPT-3.5 | GPT-5.x series |
| **Messages param** | `messages=[...]` | `input=[...]` |
| **Token limit** | `max_tokens=100` | `max_completion_tokens=100` |
| **Output access** | `result.choices[0].message` | `result.output` (iterate) |
| **Function args** | `tool_calls[0].function.arguments` | `item.arguments` |
| **Strict mode** | Supported | Supported |

### Chat Completions Example

```python
from openai import OpenAI
client = OpenAI()

result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Is 2+2=4? Reply pass or fail."}
    ],
    max_tokens=50,
)

text = result.choices[0].message.content
# Returns: "Pass" (but might be "PASS", "pass", "I pass this", etc.)
```

### Responses API Example

```python
from openai import OpenAI
client = OpenAI()

result = client.responses.create(
    model="gpt-5-mini-2025-08-07",
    input=[
        {"role": "user", "content": "Is 2+2=4? Call submit_evaluation."}
    ],
    max_completion_tokens=50,
    tools=[EVAL_TOOL],
    tool_choice={"type": "function", "name": "submit_evaluation"},
)

for item in result.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        verdict = args["verdict"]  # Guaranteed "pass" or "fail"
```

## Function Calling Schemas

### OpenAI (Responses API)

```python
EVAL_TOOL = {
    "type": "function",
    "name": "submit_evaluation",
    "description": "Submit your evaluation verdict",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail"],
                "description": "pass if correct, fail otherwise",
            }
        },
        "required": ["verdict"],
        "additionalProperties": False,
    },
}
```

### Cerebras (Chat Completions style)

```python
EVAL_TOOL = {
    "type": "function",
    "function": {  # Note: nested "function" object
        "name": "submit_evaluation",
        "description": "Submit your evaluation verdict",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["pass", "fail"],
                    "description": "pass if correct, fail otherwise",
                }
            },
            "required": ["verdict"],
        },
    },
}
```

### Gemini

```python
import google.generativeai as genai

model = genai.GenerativeModel('gemini-2.0-flash-lite')
result = model.generate_content(prompt)
text = result.text
# No native function calling in free tier - use text parsing
```

## Rate Limits

| Provider | Tier | RPM | RPD | Notes |
|----------|------|-----|-----|-------|
| OpenAI | Paid | 500+ | Unlimited | Based on usage tier |
| Gemini | Free | 15 | 1500 | RPD is bottleneck |
| Cerebras | Free | 30 | 1000 | Fast inference |

### Rate Limit Handling

```python
import time

def call_with_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            wait = 3 ** attempt  # Exponential: 1, 3, 9, 27, 81
            print(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
    raise Exception("Max retries exceeded")
```

## Output Parsing

### Chat Completions (Text)

```python
# Fragile - don't do this in production
result = client.chat.completions.create(...)
text = result.choices[0].message.content
verdict = "pass" if "pass" in text.lower() else "fail"
```

### Responses API (Function Call)

```python
# Reliable - recommended
result = client.responses.create(..., tools=[TOOL], tool_choice={...})
for item in result.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        verdict = args["verdict"]
```

### Chat Completions with Tools

```python
# Also reliable if using tools
result = client.chat.completions.create(..., tools=[TOOL], tool_choice={...})
tool_call = result.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
verdict = args["verdict"]
```

## Quick Reference

### Which API Should I Use?

| Use Case | Recommended |
|----------|-------------|
| GPT-5.x with function calling | Responses API |
| GPT-4o-mini/GPT-3.5 | Chat Completions |
| Free tier prototyping | Gemini |
| Fast inference | Cerebras |

### Which Method Should I Use?

| Use Case | Method |
|----------|--------|
| Pass/fail classification | Function calling (strict) |
| Multi-class classification | Function calling (enum) |
| Structured extraction | Function calling |
| Free-form generation | Text (no parsing needed) |

### Common Parameter Mistakes

```python
# ❌ Wrong for Responses API
client.responses.create(
    messages=[...],       # Should be "input"
    max_tokens=100,       # Should be "max_completion_tokens"
)

# ✅ Correct for Responses API
client.responses.create(
    input=[...],
    max_completion_tokens=100,
)
```
