# LLM Evaluation Pipeline

A practical implementation of [Eugene Yan's "Product Evals in Three Simple Steps"](https://eugeneyan.com/writing/product-evals/) methodology for building reliable LLM evaluators.

## Key Results

| Metric | Value | Target |
|--------|-------|--------|
| **Cohen's Kappa** | 0.615 | 0.4-0.6 (substantial) |
| **Fail Recall** | 82.4% | High (catch defects) |
| **Accuracy** | 82.0% | - |

GPT-5-mini achieves **substantial agreement** with GPT-5.1 ground truth labels, exceeding typical human inter-rater reliability (kappa 0.2-0.3).

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/nibzard/evals.git
cd evals

# Add your API keys to .env
echo "OPENAI_API_KEY=your-key" >> .env

# Start with the examples (no API key needed for metrics example)
uv run examples/kappa_calculator.py   # Understand why Kappa > accuracy

# Run your first evaluation
uv run examples/minimal_eval.py       # Simplest evaluation (1 sample)

# Verify evaluator against ground truth
uv run verify_evaluator.py --sample 50
```

## Learning Path

New to LLM evaluation? Follow this progression:

| Step | Resource | Time | What You'll Learn |
|------|----------|------|-------------------|
| 1 | `examples/kappa_calculator.py` | 5 min | Why accuracy is misleading |
| 2 | `examples/minimal_eval.py` | 10 min | Your first evaluation |
| 3 | `docs/tutorial/01-06*.md` | 1 hour | Complete methodology |
| 4 | `notebooks/*.py` (marimo) | 1 hour | Interactive exploration |
| 5 | `verify_evaluator.py` | - | Production pipeline |

```bash
# Interactive notebooks (marimo)
uv run marimo edit notebooks/01_understanding_evals.py
uv run marimo edit notebooks/02_position_bias.py
uv run marimo edit notebooks/03_kappa_intuition.py
```

## Project Structure

```
├── examples/                 # START HERE - Simple learning examples
│   ├── minimal_eval.py       # Simplest evaluation (30 lines)
│   ├── kappa_calculator.py   # Understand metrics (no API needed)
│   ├── text_vs_function.py   # Why function calling matters
│   └── position_bias_demo.py # Detect position bias
│
├── notebooks/                # Interactive marimo notebooks
│   ├── 01_understanding_evals.py
│   ├── 02_position_bias.py
│   └── 03_kappa_intuition.py
│
├── evaluators/               # Modular evaluation framework
│   ├── openai_evaluator.py   # OpenAI Responses API
│   ├── cerebras_evaluator.py # Cerebras (Llama models)
│   └── metrics.py            # Cohen's Kappa, etc.
│
├── docs/
│   ├── tutorial/             # 6-part learning progression
│   ├── glossary.md           # Key terminology
│   ├── architecture.md       # System diagrams
│   ├── api_comparison.md     # API differences
│   ├── common_mistakes.md    # Pitfalls to avoid
│   └── plots/                # Generated visualizations
│
├── eval_demo_*.py            # Provider comparison demos
├── verify_evaluator.py       # Full verification pipeline
├── plot_runs.py              # Visualize run comparisons
│
├── scripts/                  # Data generation pipeline
├── data/                     # Datasets (200 samples)
└── results/                  # Run logs (gitignored)
```

## Provider Comparison

| Provider | Model | API | Output Parsing | Position Bias |
|----------|-------|-----|----------------|---------------|
| Gemini | gemini-2.0-flash-lite | generate_content | Text matching (fragile) | Variable |
| OpenAI | gpt-4o-mini | Chat Completions | Text matching (fragile) | High ("tie") |
| OpenAI | gpt-5.1 | Responses API | Function calling (reliable) | Low (consistent) |

**Recommendation:** Use GPT-5.1 with function calling for production evaluators. The `strict: true` schema with enums guarantees valid output.

## Eugene Yan's Methodology

### 1. Label Data
- **Binary labels only** - pass/fail, not numeric scales
- **Target 50-100 failures** in 200+ samples for meaningful signal
- **Organic failures** - use weaker models, not synthetic defects

### 2. Build Evaluators
- **One evaluator per dimension** - not a "God Evaluator"
- **75/25 train/test split** - tune prompts on train, measure on test
- **Mitigate position bias** - run comparisons twice with swapped order

### 3. Measure Agreement
- **Cohen's Kappa** - accounts for chance agreement
  - 0.4-0.6 = substantial (target)
  - 0.6-0.8 = substantial to excellent
  - Human baseline often only 0.2-0.3
- **Prioritize recall on failures** - catching defects is critical

## Datasets

### v1: `data/questions.csv`
- 200 questions, 7 failures (3.5% fail rate)
- Answer model: Gemini 2.0 Flash Lite
- Too few failures for meaningful signal

### v2: `data/questions_version_2.csv`
- 200 questions, 71 failures (35.5% fail rate)
- Answer model: GPT-3.5-turbo-0125 (weaker = more failures)
- Categories designed for higher failure rates:

| Category | Fail Rate | Description |
|----------|-----------|-------------|
| precise_calculations | 60% | Math, code output, conversions |
| temporal_changing | 60% | Current events, changing facts |
| multi_part | 37% | Questions requiring multiple pieces |
| ambiguous_debatable | 26% | Contested facts |
| edge_cases | 23% | Nuanced topics, exceptions |
| trick_misconceptions | 14% | Common myths |

## Key Learnings

### 1. Function Calling > Text Parsing
```python
# Fragile - model might say "Pass", "PASS", "I think it passes"
return "pass" if "pass" in result.text.lower() else "fail"

# Reliable - enum guarantees exact values
EVAL_TOOL = {
    "strict": True,
    "parameters": {
        "properties": {
            "verdict": {"enum": ["pass", "fail"]}
        }
    }
}
```

### 2. Position Bias is Real
Run comparisons twice with swapped order. If answers differ, the model has position bias.

### 3. Smaller Models Can Evaluate
GPT-5-mini achieves kappa 0.615 against GPT-5.1 labels - substantial agreement at lower cost.

### 4. Math is Hard
`precise_calculations` category shows worst agreement (57% accuracy, kappa 0.09). Smaller models struggle with math.

## Commands Reference

```bash
# Learning examples (start here!)
uv run examples/minimal_eval.py       # Simplest evaluation
uv run examples/kappa_calculator.py   # Understand metrics (no API)
uv run examples/text_vs_function.py   # Why function calling matters
uv run examples/position_bias_demo.py # Detect position bias

# Interactive notebooks
uv run marimo edit notebooks/01_understanding_evals.py
uv run marimo edit notebooks/02_position_bias.py
uv run marimo edit notebooks/03_kappa_intuition.py

# Evaluation demos
uv run eval_demo_gpt5.py              # Recommended
uv run eval_demo_gpt4.py
uv run eval_demo_gemini.py

# Verify evaluator against ground truth
uv run verify_evaluator.py                              # Full dataset
uv run verify_evaluator.py --sample 50                  # Sample 50 items
uv run verify_evaluator.py --model gpt-5-nano-2025-08-07  # Test different model
uv run verify_evaluator.py --category precise_calculations  # Filter by category

# Visualize evaluation runs
uv run plot_runs.py                   # Save to docs/plots/
uv run plot_runs.py --last 5          # Only last 5 runs

# Data generation pipeline
uv run scripts/generate_hard_questions.py  # Generate questions + answers
uv run scripts/label_responses.py          # Label with GPT-5.1
```

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/tutorial/`](docs/tutorial/) | 6-part learning progression |
| [`docs/glossary.md`](docs/glossary.md) | Key terminology (Kappa, position bias, etc.) |
| [`docs/architecture.md`](docs/architecture.md) | System diagrams and data flow |
| [`docs/common_mistakes.md`](docs/common_mistakes.md) | 10 pitfalls to avoid |
| [`docs/api_comparison.md`](docs/api_comparison.md) | API differences between providers |
| [`EVAL_PROVIDERS.md`](EVAL_PROVIDERS.md) | Detailed provider comparison |
| [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md) | API patterns and gotchas |

## References

- [Eugene Yan: Product Evals in Three Simple Steps](https://eugeneyan.com/writing/product-evals/)
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [Cohen's Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)

## License

MIT
