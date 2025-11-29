#!/usr/bin/env python3
"""
Position Bias Detection Demo
=============================
Understand why LLMs can prefer responses based on ORDER, not quality.

Run: uv run examples/position_bias_demo.py

What you'll learn:
- Position bias: models favor first or last option regardless of content
- How to detect it: run comparison twice with swapped order
- How to handle it: if answers differ, return "tie"

This is critical for A/B testing and comparative evaluations.
"""

from openai import OpenAI
import json

# Two responses to compare - Response B is clearly better
QUESTION = "Explain what a variable is in programming."
RESPONSE_A = "A variable stores data."  # Too brief
RESPONSE_B = "A variable is a named container that stores a value in memory. You can change its value during program execution. For example, `age = 25` creates a variable called 'age' with value 25."  # Better


def compare_responses(response_1: str, response_2: str, labels: tuple[str, str]) -> str:
    """
    Compare two responses and return which is better.

    Args:
        response_1: First response shown
        response_2: Second response shown
        labels: Tuple of labels (e.g., ("A", "B"))

    Returns:
        "A", "B", or "tie"
    """
    client = OpenAI()

    prompt = f"""Compare these two responses to the question and decide which is better.

Question: {QUESTION}

Response {labels[0]}:
{response_1}

Response {labels[1]}:
{response_2}

Which response is better? Call submit_comparison with your choice."""

    tool = {
        "type": "function",
        "name": "submit_comparison",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "winner": {
                    "type": "string",
                    "enum": [labels[0], labels[1], "tie"],
                }
            },
            "required": ["winner"],
            "additionalProperties": False,
        },
    }

    result = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=[{"role": "user", "content": prompt}],
        tools=[tool],
        tool_choice={"type": "function", "name": "submit_comparison"},
        max_completion_tokens=50,
    )

    for item in result.output:
        if item.type == "function_call":
            return json.loads(item.arguments)["winner"]

    raise ValueError("No function call")


def detect_position_bias() -> dict:
    """
    Run comparison twice with swapped order to detect position bias.

    If the model gives different answers when order is swapped,
    it's showing position bias - return "tie" instead.
    """
    # First comparison: A first, B second
    print("  Round 1: A shown first, B shown second")
    result_1 = compare_responses(RESPONSE_A, RESPONSE_B, ("A", "B"))
    print(f"    Winner: {result_1}")

    # Second comparison: B first, A second (swapped!)
    print("  Round 2: B shown first, A shown second")
    result_2 = compare_responses(RESPONSE_B, RESPONSE_A, ("B", "A"))
    print(f"    Winner: {result_2}")

    # Analyze results
    if result_1 == result_2:
        # Same winner both times = genuine preference
        final = result_1
        has_bias = False
    else:
        # Different answers = position bias detected!
        final = "tie"
        has_bias = True

    return {
        "round_1": result_1,
        "round_2": result_2,
        "final": final,
        "position_bias_detected": has_bias,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("POSITION BIAS DETECTION")
    print("=" * 60)
    print()
    print("What is position bias?")
    print("  LLMs often prefer responses based on their POSITION in the prompt")
    print("  (first vs last) rather than their actual quality.")
    print()
    print("Why it matters:")
    print("  - A/B tests become unreliable")
    print("  - Comparative evaluations give wrong answers")
    print("  - You might ship worse features!")
    print()
    print("How to detect it:")
    print("  1. Run comparison: A first, B second")
    print("  2. Run comparison: B first, A second (swapped)")
    print("  3. If answers differ -> position bias! Return 'tie'")
    print()
    print("-" * 60)
    print()
    print("Responses being compared:")
    print(f"  Response A: '{RESPONSE_A}'")
    print(f"  Response B: '{RESPONSE_B[:60]}...'")
    print()
    print("(Response B is clearly more helpful)")
    print()
    print("-" * 60)
    print()
    print("Running position bias detection...")
    print()

    results = detect_position_bias()

    print()
    print("-" * 60)
    print("RESULTS:")
    print(f"  Round 1 winner: {results['round_1']}")
    print(f"  Round 2 winner: {results['round_2']}")
    print(f"  Position bias: {'YES' if results['position_bias_detected'] else 'NO'}")
    print(f"  Final verdict: {results['final']}")
    print()

    if results["position_bias_detected"]:
        print("  Position bias was detected!")
        print("  The model gave different answers based on order, not quality.")
        print("  We return 'tie' to avoid a misleading result.")
    else:
        print("  No position bias detected.")
        print("  The model consistently preferred the same response.")
    print()
    print("=" * 60)
