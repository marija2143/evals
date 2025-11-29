#!/usr/bin/env python3
"""
Minimal Evaluation Example
==========================
This is the simplest possible LLM evaluation - just 1 sample, 1 metric.

Run: uv run examples/minimal_eval.py

What you'll learn:
- How to evaluate a response using function calling
- Why function calling beats text parsing
- The basic structure of an evaluation

See docs/glossary.md for terminology.
"""

from openai import OpenAI
import json

# =============================================================================
# STEP 1: Define what we're evaluating
# =============================================================================
# This is our "ground truth" - a sample with a known correct label
SAMPLE = {
    "question": "What is 2 + 2?",
    "response": "The answer is 4.",
    "expected": "pass",  # We know this response is correct
}

# =============================================================================
# STEP 2: Define our evaluation tool
# =============================================================================
# We use "strict" mode to guarantee the model returns exactly "pass" or "fail"
# Without this, we'd have to parse text like "I think it passes" - fragile!
EVAL_TOOL = {
    "type": "function",
    "name": "submit_evaluation",
    "description": "Submit your evaluation verdict",
    "strict": True,  # <-- Enforces schema compliance
    "parameters": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail"],  # <-- Only these values allowed
                "description": "pass if response is correct, fail otherwise",
            }
        },
        "required": ["verdict"],
        "additionalProperties": False,  # <-- Required for strict mode
    },
}

# =============================================================================
# STEP 3: Run the evaluation
# =============================================================================
def evaluate_sample(sample: dict) -> str:
    """
    Evaluate a single question/response pair.

    Returns: "pass" or "fail" (guaranteed by strict schema)
    """
    client = OpenAI()

    # Build the prompt - tell the model exactly what to do
    prompt = f"""Evaluate if this response correctly answers the question.

Question: {sample["question"]}
Response: {sample["response"]}

Call submit_evaluation with verdict="pass" if correct, "fail" if incorrect."""

    # Call the model with function calling
    result = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=[{"role": "user", "content": prompt}],
        tools=[EVAL_TOOL],
        tool_choice={"type": "function", "name": "submit_evaluation"},
        max_completion_tokens=50,
    )

    # Extract the verdict from the function call
    for item in result.output:
        if item.type == "function_call":
            args = json.loads(item.arguments)
            return args["verdict"]

    raise ValueError("No function call in response")


# =============================================================================
# STEP 4: Check our result
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("MINIMAL EVALUATION EXAMPLE")
    print("=" * 50)
    print()
    print(f"Question: {SAMPLE['question']}")
    print(f"Response: {SAMPLE['response']}")
    print(f"Expected: {SAMPLE['expected']}")
    print()

    verdict = evaluate_sample(SAMPLE)

    print(f"Predicted: {verdict}")
    print(f"Match: {'Yes' if verdict == SAMPLE['expected'] else 'No'}")
    print()
    print("=" * 50)
    print("Next steps:")
    print("  - See examples/text_vs_function.py for why function calling matters")
    print("  - See examples/kappa_calculator.py to understand metrics")
    print("  - Run 'uv run verify_evaluator.py --sample 10' for real evaluation")
