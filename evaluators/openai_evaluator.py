"""
OpenAI Evaluator
================

Evaluator using OpenAI's Responses API with function calling.

This is the RECOMMENDED approach for production evaluations because:
1. Function calling guarantees structured output
2. Strict schema ensures only "pass" or "fail" values
3. No fragile text parsing required

See:
    - docs/tutorial/03_function_calling.md
    - docs/api_comparison.md
    - examples/text_vs_function.py

Example:
    from evaluators import OpenAIEvaluator

    evaluator = OpenAIEvaluator(model="gpt-5-mini-2025-08-07")
    verdict, reasoning = evaluator.evaluate(
        question="What is the capital of France?",
        response="Paris is the capital of France."
    )
    print(verdict)  # "pass"
"""

import os
import json
import time
import random
from dotenv import load_dotenv
from openai import OpenAI

from .base import BaseEvaluator

load_dotenv()


# Tool definition with strict schema
# See docs/glossary.md#strict-schema for explanation
EVAL_TOOL = {
    "type": "function",
    "name": "submit_evaluation",
    "description": "Submit the evaluation result for a question/response pair",
    "strict": True,  # Enforces schema compliance
    "parameters": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail"],  # Only these values allowed
                "description": "pass if the response correctly answers the question, fail otherwise",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation for the verdict (1-2 sentences)",
            },
        },
        "required": ["verdict", "reasoning"],
        "additionalProperties": False,  # Required for strict mode
    },
}


class OpenAIEvaluator(BaseEvaluator):
    """
    Evaluator using OpenAI's Responses API.

    Uses function calling with strict schema to guarantee valid output.
    This is the recommended approach for production evaluations.

    Attributes:
        model: The OpenAI model to use (e.g., "gpt-5-mini-2025-08-07")
        retries: Number of retries on transient failures
        client: OpenAI client instance

    Example:
        evaluator = OpenAIEvaluator()  # Uses default model
        verdict, reasoning = evaluator.evaluate(question, response)
    """

    # Models that work with this evaluator
    SUPPORTED_MODELS = {
        "gpt-5.1-2025-11-13",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07",
    }

    def __init__(self, model: str = "gpt-5-mini-2025-08-07", retries: int = 3):
        """
        Initialize the OpenAI evaluator.

        Args:
            model: OpenAI model ID (default: gpt-5-mini-2025-08-07)
            retries: Number of retries on failure (default: 3)

        Raises:
            ValueError: If OPENAI_API_KEY is not set
        """
        super().__init__(model=model, retries=retries)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)

    def evaluate(self, question: str, response: str) -> tuple[str, str]:
        """
        Evaluate a question/response pair.

        Uses the Responses API with function calling for reliable output.

        Args:
            question: The question that was asked
            response: The AI-generated response to evaluate

        Returns:
            Tuple of (verdict, reasoning):
            - verdict: "pass" or "fail" (guaranteed by strict schema)
            - reasoning: Brief explanation

        Note:
            Implements exponential backoff on transient failures.
            See docs/common_mistakes.md#6-not-handling-rate-limits
        """
        prompt = self.get_prompt(question, response)

        for attempt in range(self.retries):
            try:
                # Call the Responses API (not Chat Completions!)
                # Key differences documented in docs/api_comparison.md
                result = self.client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": prompt}],  # "input" not "messages"
                    tools=[EVAL_TOOL],
                    tool_choice={"type": "function", "name": "submit_evaluation"},
                )

                # Extract verdict from function call
                # The model is forced to call submit_evaluation due to tool_choice
                for item in result.output:
                    if item.type == "function_call" and item.name == "submit_evaluation":
                        args = json.loads(item.arguments)
                        return args.get("verdict", "fail"), args.get("reasoning", "")

                return "fail", "No function call found in response"

            except Exception as e:
                if attempt < self.retries - 1:
                    # Exponential backoff with jitter
                    delay = (2**attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    return "fail", f"Error after {self.retries} attempts: {e}"

        return "fail", "Max retries exceeded"
