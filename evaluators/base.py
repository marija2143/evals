"""
Base Evaluator Interface
========================

All evaluators implement this interface for consistency.

Key concepts:
    - evaluate() returns (verdict, reasoning) tuple
    - verdict is always "pass" or "fail" (binary, per Eugene Yan)
    - reasoning provides context for debugging

See docs/glossary.md for terminology.
"""

from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """
    Abstract base class for LLM evaluators.

    All evaluators must implement the evaluate() method, which takes a
    question and response and returns a (verdict, reasoning) tuple.

    Example:
        evaluator = SomeEvaluator(model="gpt-5-mini")
        verdict, reasoning = evaluator.evaluate(
            question="What is 2+2?",
            response="The answer is 4."
        )
        # verdict: "pass" or "fail"
        # reasoning: "The response correctly answers..."
    """

    def __init__(self, model: str, retries: int = 3):
        """
        Initialize the evaluator.

        Args:
            model: The model ID to use for evaluation
            retries: Number of retries on transient failures (default: 3)
        """
        self.model = model
        self.retries = retries

    @abstractmethod
    def evaluate(self, question: str, response: str) -> tuple[str, str]:
        """
        Evaluate whether a response correctly answers a question.

        Args:
            question: The question that was asked
            response: The AI-generated response to evaluate

        Returns:
            Tuple of (verdict, reasoning):
            - verdict: "pass" if correct, "fail" otherwise
            - reasoning: Brief explanation of the verdict

        Note:
            This method should use function calling with strict schema
            to guarantee valid output. See docs/tutorial/03_function_calling.md
        """
        pass

    def get_prompt(self, question: str, response: str) -> str:
        """
        Generate the evaluation prompt.

        Override this method to customize the evaluation criteria.

        Args:
            question: The question that was asked
            response: The AI-generated response to evaluate

        Returns:
            The prompt to send to the LLM
        """
        return f"""You are evaluating whether an AI response correctly and helpfully answers a question.

Question: {question}

Response: {response}

Evaluate the response on these criteria:
1. Is the response factually correct?
2. Does it actually answer the question asked?
3. Is it complete and not misleading?

Use the submit_evaluation tool to provide your verdict:
- "pass" if the response is correct and helpful
- "fail" if the response is wrong, incomplete, or misleading"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model}')"
