#!/usr/bin/env python3
"""
Text Parsing vs Function Calling
================================
This example demonstrates WHY function calling is essential for reliable evals.

Run: uv run examples/text_vs_function.py

What you'll learn:
- Text parsing is fragile and fails in many ways
- Function calling with strict schema guarantees valid output
- The difference between Chat Completions and Responses API
"""

from openai import OpenAI
import json

# Sample to evaluate
SAMPLE = {
    "question": "What is the capital of France?",
    "response": "Paris is the capital of France.",
}


def evaluate_with_text_parsing() -> tuple[str, str]:
    """
    BAD APPROACH: Parse verdict from free-form text.

    This is fragile because the model might say:
    - "Pass" or "PASS" or "pass" (case variations)
    - "I would pass this response" (embedded in sentence)
    - "The response passes the test" (verb form)
    - "It's a passing grade" (adjective form)
    - "Verdict: Pass." (with punctuation)
    """
    client = OpenAI()

    prompt = f"""Evaluate if this response correctly answers the question.
Reply with just "pass" or "fail".

Question: {SAMPLE["question"]}
Response: {SAMPLE["response"]}"""

    # Using Chat Completions API (older, but shows the problem)
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
    )

    raw_text = result.choices[0].message.content
    # This parsing is fragile!
    parsed = "pass" if "pass" in raw_text.lower() else "fail"

    return raw_text, parsed


def evaluate_with_function_calling() -> tuple[str, str]:
    """
    GOOD APPROACH: Use function calling with strict schema.

    The model MUST return exactly "pass" or "fail" - no parsing needed.
    """
    client = OpenAI()

    prompt = f"""Evaluate if this response correctly answers the question.

Question: {SAMPLE["question"]}
Response: {SAMPLE["response"]}

Call submit_evaluation with your verdict."""

    # Tool with strict schema
    tool = {
        "type": "function",
        "name": "submit_evaluation",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["pass", "fail"],
                }
            },
            "required": ["verdict"],
            "additionalProperties": False,
        },
    }

    # Using Responses API (newer, recommended for GPT-5.x)
    result = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=[{"role": "user", "content": prompt}],
        tools=[tool],
        tool_choice={"type": "function", "name": "submit_evaluation"},
        max_completion_tokens=50,
    )

    # Extract from function call - guaranteed valid
    for item in result.output:
        if item.type == "function_call":
            raw = item.arguments
            parsed = json.loads(raw)["verdict"]
            return raw, parsed

    raise ValueError("No function call")


# =============================================================================
# Demonstrate failure cases
# =============================================================================

TRICKY_RESPONSES = [
    "I would pass this response as correct.",
    "PASS",
    "Pass.",
    "The response passes the accuracy test.",
    "Verdict: pass (with high confidence)",
    "This is a passing answer.",
]


def parse_text(text: str) -> str:
    """Naive text parsing - shows why this breaks."""
    return "pass" if "pass" in text.lower() else "fail"


if __name__ == "__main__":
    print("=" * 60)
    print("TEXT PARSING VS FUNCTION CALLING")
    print("=" * 60)
    print()

    # Show problematic parsing
    print("PROBLEM: Text parsing is fragile")
    print("-" * 40)
    for response in TRICKY_RESPONSES:
        parsed = parse_text(response)
        # All of these SHOULD parse as "pass", but the logic is messy
        print(f"  '{response[:40]}...' -> {parsed}")
    print()
    print("  All above parse correctly, but what about:")
    print("    'The bypass fails' -> ", parse_text("The bypass fails"))
    print("    'compassionate response' -> ", parse_text("compassionate response"))
    print("  False positives from 'pass' substring!")
    print()

    # Run actual comparisons
    print("COMPARISON: Actual API calls")
    print("-" * 40)

    print("\n1. Text Parsing (Chat Completions API):")
    raw_text, parsed_text = evaluate_with_text_parsing()
    print(f"   Raw output: '{raw_text}'")
    print(f"   Parsed as: '{parsed_text}'")

    print("\n2. Function Calling (Responses API):")
    raw_fn, parsed_fn = evaluate_with_function_calling()
    print(f"   Raw output: {raw_fn}")
    print(f"   Parsed as: '{parsed_fn}' (guaranteed valid)")

    print()
    print("=" * 60)
    print("KEY TAKEAWAY:")
    print("  Function calling + strict schema = reliable evaluation")
    print("  Text parsing = unpredictable failures at scale")
    print("=" * 60)
