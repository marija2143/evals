#!/usr/bin/env python3
"""
Position Bias Detection - Interactive Notebook
===============================================
Run with: uv run marimo edit notebooks/02_position_bias.py

This notebook demonstrates position bias in LLM comparisons
and how to detect and handle it.
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Position Bias in LLM Comparisons

        When comparing two responses, LLMs can prefer one based on its **position**
        (first vs last) rather than its actual quality.

        **What you'll learn:**
        1. What position bias looks like
        2. How to detect it
        3. How to handle it in production

        This notebook includes an optional live API experiment.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        ## The Problem

        Imagine comparing two responses:

        ```
        Prompt: "Which response is better?"

        Response A: [Brief answer]
        Response B: [Detailed explanation]
        ```

        A model with position bias might prefer:
        - **Primacy bias**: Always the first option (A)
        - **Recency bias**: Always the last option (B)

        This makes A/B tests unreliable!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Simulated Position Bias

        Let's simulate what position bias looks like with synthetic data.
        """
    )
    return


@app.cell
def _():
    import random

    def simulate_comparison(response_a: str, response_b: str, first_label: str, bias_strength: float = 0.3):
        """
        Simulate a comparison with position bias.

        Args:
            response_a: First response content (ignored in biased model)
            response_b: Second response content (ignored in biased model)
            first_label: Label of first-shown response ("A" or "B")
            bias_strength: Probability of choosing first option regardless of quality

        Returns:
            Winner label ("A", "B", or "tie")
        """
        # True quality (B is actually better in our simulation)
        true_winner = "B"

        # With bias_strength probability, prefer first position
        if random.random() < bias_strength:
            return first_label  # Biased toward first position
        else:
            return true_winner  # Correct answer

    return random, simulate_comparison


@app.cell
def _(mo):
    bias_slider = mo.ui.slider(
        0, 1, value=0.4, step=0.1,
        label="Position bias strength (0 = no bias, 1 = always prefer first)"
    )
    trials_slider = mo.ui.slider(
        10, 100, value=50, step=10,
        label="Number of comparison trials"
    )

    mo.vstack([bias_slider, trials_slider])
    return bias_slider, trials_slider


@app.cell
def _(bias_slider, mo, random, simulate_comparison, trials_slider):
    random.seed(42)  # For reproducibility

    bias = bias_slider.value
    n_trials = trials_slider.value

    # Run trials
    results = {"consistent": 0, "inconsistent": 0}

    for _ in range(n_trials):
        # Round 1: A shown first
        result1 = simulate_comparison("brief", "detailed", "A", bias)

        # Round 2: B shown first (swapped)
        result2 = simulate_comparison("detailed", "brief", "B", bias)

        if result1 == result2:
            results["consistent"] += 1
        else:
            results["inconsistent"] += 1

    consistency_rate = results["consistent"] / n_trials
    bias_detection_rate = results["inconsistent"] / n_trials

    mo.md(
        f"""
        ## Simulation Results

        **Settings:**
        - Position bias strength: {bias:.0%}
        - Trials: {n_trials}

        **Results:**
        | Outcome | Count | Rate |
        |---------|-------|------|
        | Consistent (same winner both rounds) | {results['consistent']} | {consistency_rate:.1%} |
        | Inconsistent (different winners → bias detected) | {results['inconsistent']} | {bias_detection_rate:.1%} |

        **Interpretation:**
        - Inconsistent results indicate position bias was detected
        - At {bias:.0%} bias strength, we detect it {bias_detection_rate:.1%} of the time
        - In production, return "tie" for inconsistent comparisons
        """
    )
    return (
        bias,
        bias_detection_rate,
        consistency_rate,
        n_trials,
        result1,
        result2,
        results,
    )


@app.cell
def _(mo):
    mo.md(
        """
        ## Detection Algorithm

        The fix is simple: run every comparison **twice** with swapped order.

        ```python
        def compare_with_bias_detection(response_a, response_b):
            # Round 1: Original order
            result1 = compare(response_a, response_b, labels=("A", "B"))

            # Round 2: Swapped order
            result2 = compare(response_b, response_a, labels=("B", "A"))

            # Check consistency
            if result1 == result2:
                return result1  # Genuine preference
            else:
                return "tie"    # Position bias detected
        ```

        This doubles API costs but ensures reliable results.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Live API Experiment (Optional)

        Run an actual position bias test with the OpenAI API.

        **Requirements:**
        - Set `OPENAI_API_KEY` environment variable
        - This will make 2 API calls
        """
    )
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run Live Position Bias Test")
    run_button
    return (run_button,)


@app.cell
def _(mo, run_button):
    import os
    import json

    if not run_button.value:
        mo.md("*Click the button above to run the live test*")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            mo.md("**Error:** OPENAI_API_KEY not set. Please set it in your environment.")
        else:
            try:
                from openai import OpenAI

                client = OpenAI()

                question = "Explain what a variable is in programming."
                response_a = "A variable stores data."
                response_b = "A variable is a named container that stores a value in memory. You can change its value during program execution. For example, `age = 25` creates a variable called 'age' with value 25."

                tool = {
                    "type": "function",
                    "name": "submit_comparison",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "winner": {
                                "type": "string",
                                "enum": ["A", "B", "tie"],
                            }
                        },
                        "required": ["winner"],
                        "additionalProperties": False,
                    },
                }

                def compare(resp_1, resp_2, labels):
                    prompt = f"""Compare these two responses and decide which is better.

Question: {question}

Response {labels[0]}:
{resp_1}

Response {labels[1]}:
{resp_2}

Which response is better? Call submit_comparison with your choice."""

                    # Update tool enum dynamically
                    tool_copy = json.loads(json.dumps(tool))
                    tool_copy["parameters"]["properties"]["winner"]["enum"] = [labels[0], labels[1], "tie"]

                    result = client.responses.create(
                        model="gpt-5-mini-2025-08-07",
                        input=[{"role": "user", "content": prompt}],
                        tools=[tool_copy],
                        tool_choice={"type": "function", "name": "submit_comparison"},
                        max_completion_tokens=50,
                    )

                    for item in result.output:
                        if item.type == "function_call":
                            return json.loads(item.arguments)["winner"]
                    return "error"

                # Round 1: A first
                result1 = compare(response_a, response_b, ("A", "B"))

                # Round 2: B first (swapped)
                result2 = compare(response_b, response_a, ("B", "A"))

                # Determine outcome
                if result1 == result2:
                    final = result1
                    bias_detected = False
                else:
                    final = "tie"
                    bias_detected = True

                mo.md(
                    f"""
                    ## Live Test Results

                    **Responses compared:**
                    - **A (brief):** "{response_a}"
                    - **B (detailed):** "{response_b[:50]}..."

                    **Results:**
                    | Round | Order | Winner |
                    |-------|-------|--------|
                    | 1 | A first, B second | {result1} |
                    | 2 | B first, A second | {result2} |

                    **Analysis:**
                    - Position bias detected: **{'YES' if bias_detected else 'NO'}**
                    - Final verdict: **{final}**

                    {'⚠️ Different answers in each round indicates position bias. We return "tie" to avoid a misleading result.' if bias_detected else '✓ Consistent answers - the model genuinely prefers this response.'}
                    """
                )

            except Exception as e:
                mo.md(f"**Error:** {e}")
    return api_key, client, compare, json, os, question, response_a, response_b, tool


@app.cell
def _(mo):
    mo.md(
        """
        ## Key Takeaways

        1. **Position bias is real** - LLMs can prefer responses based on order

        2. **Detection is simple** - Run comparison twice with swapped order

        3. **Handle inconsistency** - Return "tie" when results differ

        4. **Track bias rate** - High rates (>20%) indicate problems:
           - Responses too similar to distinguish
           - Model has strong position preference
           - Prompt needs improvement

        5. **Newer models are better** - GPT-5.x has less position bias than older models

        ---

        **Next:** Open `notebooks/03_kappa_intuition.py` for Kappa visualizations
        """
    )
    return


if __name__ == "__main__":
    app.run()
