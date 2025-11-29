"""
Cerebras Evaluator
==================

Evaluator using Cerebras' Chat Completions API (OpenAI-compatible).

Cerebras provides fast inference for open-source models. This evaluator
uses their function calling capability for structured output.

Note: Tool schema differs from OpenAI - the "function" key is nested.
See docs/api_comparison.md for differences.

Example:
    from evaluators import CerebrasEvaluator

    evaluator = CerebrasEvaluator(model="llama3.1-8b")
    verdict, reasoning = evaluator.evaluate(question, response)
"""

import os
import json
import time
import random
from dotenv import load_dotenv
from openai import OpenAI

from .base import BaseEvaluator

load_dotenv()


# Tool definition for Cerebras (different structure from OpenAI!)
# Note: "strict" goes INSIDE the "function" object
# See: https://inference-docs.cerebras.ai/capabilities/tool-use
EVAL_TOOL = {
    "type": "function",
    "function": {  # <-- Nested "function" object (different from OpenAI)
        "name": "submit_evaluation",
        "strict": True,  # <-- Goes here, not at top level
        "description": "Submit the evaluation result for a question/response pair",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["pass", "fail"],
                    "description": "pass if the response correctly answers the question, fail otherwise",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation for the verdict (1-2 sentences)",
                },
            },
            "required": ["verdict", "reasoning"],
            "additionalProperties": False,
        },
    },
}


class CerebrasEvaluator(BaseEvaluator):
    """
    Evaluator using Cerebras' OpenAI-compatible API.

    Cerebras provides fast inference for open-source models like Llama.
    Uses function calling for structured output.

    Attributes:
        model: The Cerebras model to use (e.g., "llama3.1-8b")
        retries: Number of retries on transient failures
        client: OpenAI-compatible client for Cerebras

    Supported models:
        - llama3.1-8b
        - llama-3.3-70b
        - qwen-3-32b
    """

    # Models hosted on Cerebras
    SUPPORTED_MODELS = {"llama3.1-8b", "llama-3.3-70b", "gpt-oss-120b", "qwen-3-32b"}

    def __init__(self, model: str = "llama3.1-8b", retries: int = 5):
        """
        Initialize the Cerebras evaluator.

        Args:
            model: Cerebras model ID (default: llama3.1-8b)
            retries: Number of retries on failure (default: 5, more for rate limits)

        Raises:
            ValueError: If CEREBRAS_API_KEY is not set
        """
        super().__init__(model=model, retries=retries)

        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY environment variable not set")

        # OpenAI-compatible client with Cerebras base URL
        self.client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")

    def evaluate(self, question: str, response: str) -> tuple[str, str]:
        """
        Evaluate a question/response pair.

        Uses Chat Completions API with function calling.

        Args:
            question: The question that was asked
            response: The AI-generated response to evaluate

        Returns:
            Tuple of (verdict, reasoning):
            - verdict: "pass" or "fail"
            - reasoning: Brief explanation

        Note:
            Uses longer backoff for rate limits (Cerebras free tier is limited).
        """
        prompt = self.get_prompt(question, response)

        for attempt in range(self.retries):
            try:
                # Chat Completions API (different from OpenAI Responses API)
                result = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an evaluation assistant. Always use the submit_evaluation tool to provide your verdict.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tools=[EVAL_TOOL],
                    tool_choice={"type": "function", "function": {"name": "submit_evaluation"}},
                    temperature=0,  # Deterministic for evaluation
                )

                message = result.choices[0].message

                # Check for tool calls
                if message.tool_calls:
                    tool_call = message.tool_calls[0]
                    if tool_call.function.name == "submit_evaluation":
                        args = json.loads(tool_call.function.arguments)
                        return args.get("verdict", "fail"), args.get("reasoning", "")[:200]

                # Fallback: parse text if no tool call (shouldn't happen)
                if message.content:
                    text = message.content.strip().upper()
                    if "PASS" in text[:20]:
                        return "pass", "Fallback text parse: " + message.content[:100]
                    else:
                        return "fail", "Fallback text parse: " + message.content[:100]

                return "fail", "No tool call or content in response"

            except Exception as e:
                error_str = str(e)

                # Rate limit handling with longer backoff
                if "429" in error_str or "rate" in error_str.lower():
                    if attempt < self.retries - 1:
                        delay = (3**attempt) + random.uniform(1, 3)  # Longer for rate limits
                        time.sleep(delay)
                        continue

                # Standard backoff for other errors
                if attempt < self.retries - 1:
                    delay = (2**attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    return "fail", f"Error after {self.retries} attempts: {error_str[:100]}"

        return "fail", "Max retries exceeded"
