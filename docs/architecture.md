# Architecture

Visual overview of the LLM evaluation pipeline.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LLM EVALUATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────────┐     ┌──────────────────────┐                │
│   │  1. GENERATE DATA    │     │  2. LABEL DATA       │                │
│   │                      │     │                      │                │
│   │  GPT-5.1 generates   │────▶│  GPT-5.1 labels      │                │
│   │  hard questions      │     │  pass/fail           │                │
│   │                      │     │  (ground truth)      │                │
│   │  GPT-3.5 answers     │     │                      │                │
│   │  (weak model =       │     │                      │                │
│   │   more failures)     │     │                      │                │
│   └──────────────────────┘     └──────────┬───────────┘                │
│                                           │                             │
│                                           ▼                             │
│                              ┌──────────────────────┐                   │
│                              │  data/questions.csv  │                   │
│                              │  (200 samples with   │                   │
│                              │   ground truth)      │                   │
│                              └──────────┬───────────┘                   │
│                                         │                               │
│                                         ▼                               │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │  3. VERIFY EVALUATOR                                         │     │
│   │                                                              │     │
│   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │     │
│   │  │ GPT-5-mini  │    │ GPT-5-nano  │    │  Cerebras   │      │     │
│   │  │ (balanced)  │    │ (cheapest)  │    │ (alt)       │      │     │
│   │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘      │     │
│   │         │                  │                  │              │     │
│   │         └──────────────────┴──────────────────┘              │     │
│   │                            │                                 │     │
│   │                            ▼                                 │     │
│   │              Compare predictions vs ground truth             │     │
│   │              Calculate: Accuracy, Kappa, Fail Recall         │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                         │                               │
│                                         ▼                               │
│                              ┌──────────────────────┐                   │
│                              │  results/runs.jsonl  │                   │
│                              └──────────┬───────────┘                   │
│                                         │                               │
│                                         ▼                               │
│                              ┌──────────────────────┐                   │
│                              │  4. VISUALIZE        │                   │
│                              │  plot_runs.py        │                   │
│                              │  → docs/plots/*.png  │                   │
│                              └──────────────────────┘                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Generation Pipeline

```mermaid
flowchart TD
    subgraph Generate["1. Generate Hard Questions"]
        A[scripts/generate_hard_questions.py]
        A1[GPT-5.1: Generate questions]
        A2[GPT-3.5: Answer questions]
        A --> A1 --> A2
    end

    subgraph Label["2. Label Responses"]
        B[scripts/label_responses.py]
        B1[GPT-5.1 with function calling]
        B2["verdict: pass | fail"]
        B --> B1 --> B2
    end

    subgraph Data["Ground Truth Dataset"]
        C[data/questions_version_2.csv]
        C1["id, question, response, label, category"]
    end

    A2 --> |CSV with empty labels| B
    B2 --> |CSV with labels| C
    C1 --> |"200 samples, 35.5% fail rate"| D[Ready for verification]
```

## Verification Flow

```mermaid
flowchart LR
    subgraph Input
        A[data/questions_version_2.csv]
    end

    subgraph Evaluator["Evaluator Model"]
        B[GPT-5-mini]
        C[GPT-5-nano]
        D[Cerebras llama3.1-8b]
    end

    subgraph Process
        E["For each sample:"]
        E1["1. Send question + response"]
        E2["2. Get verdict via function call"]
        E3["3. Compare to ground truth"]
    end

    subgraph Metrics
        F[Confusion Matrix]
        G[Cohen's Kappa]
        H[Fail Recall]
        I[Per-category breakdown]
    end

    subgraph Output
        J[results/runs.jsonl]
        K[Console report]
    end

    A --> B & C & D
    B & C & D --> E
    E --> E1 --> E2 --> E3
    E3 --> F & G & H & I
    F & G & H & I --> J & K
```

## Function Calling Flow

```mermaid
sequenceDiagram
    participant App as Your Code
    participant API as OpenAI API
    participant Model as GPT-5.x

    App->>API: responses.create(tools=[EVAL_TOOL])
    API->>Model: Process prompt + schema
    Model->>Model: Decide: pass or fail?
    Model->>API: function_call: {"verdict": "pass"}
    API->>App: result.output
    App->>App: json.loads(item.arguments)
    Note over App: Guaranteed valid value!
```

## Position Bias Detection

```mermaid
flowchart TD
    subgraph Round1["Round 1: Original Order"]
        A1["Compare A vs B"]
        A2["A shown first"]
        A3["Result: B wins"]
    end

    subgraph Round2["Round 2: Swapped Order"]
        B1["Compare B vs A"]
        B2["B shown first"]
        B3["Result: B wins"]
    end

    subgraph Decision
        C1{Same winner?}
        C2["No bias: Return winner"]
        C3["Bias detected: Return tie"]
    end

    A1 --> A2 --> A3
    B1 --> B2 --> B3
    A3 --> C1
    B3 --> C1
    C1 -->|Yes| C2
    C1 -->|No| C3
```

## File Relationships

```
evals/
├── examples/                    # Start here! Simple examples
│   ├── minimal_eval.py         # 1 sample, 1 metric
│   ├── text_vs_function.py     # Why function calling matters
│   ├── kappa_calculator.py     # Understand metrics (no API)
│   └── position_bias_demo.py   # Detect order effects
│
├── eval_demo_*.py              # Three implementations compared
│   ├── eval_demo_gemini.py     # Free tier, text parsing
│   ├── eval_demo_gpt4.py       # Chat Completions, text parsing
│   └── eval_demo_gpt5.py       # Responses API, function calling ✓
│
├── verify_evaluator.py         # Full verification pipeline
│
├── scripts/                    # Data generation
│   ├── generate_hard_questions.py  # Create challenging questions
│   ├── label_responses.py          # Add ground truth labels
│   └── generate_answers.py         # Gemini answer generation
│
├── data/
│   └── questions_version_2.csv # 200 samples, 35.5% fail rate
│
├── results/
│   └── runs.jsonl              # Historical run logs
│
├── docs/
│   ├── plots/                  # Visualizations
│   ├── tutorial/               # Learning progression
│   ├── architecture.md         # This file
│   ├── glossary.md             # Key terms
│   ├── api_comparison.md       # API differences
│   └── common_mistakes.md      # Pitfalls to avoid
│
└── notebooks/                  # Interactive marimo notebooks
    ├── 01_understanding_evals.py
    ├── 02_position_bias.py
    └── 03_kappa_intuition.py
```

## Learning Path

```
Start Here
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  1. CONCEPTS (30 min)                                       │
│     docs/glossary.md          - Key terminology             │
│     docs/tutorial/01_*.md     - Why evals matter            │
└──────────────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────┐
│  2. EXAMPLES (1 hour)                                       │
│     examples/minimal_eval.py     - Simplest evaluation      │
│     examples/kappa_calculator.py - Metrics (no API)         │
│     examples/text_vs_function.py - Why function calling     │
└──────────────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────┐
│  3. NOTEBOOKS (1 hour)                                      │
│     notebooks/01_*.py  - Interactive experiments            │
│     notebooks/02_*.py  - Position bias exploration          │
│     notebooks/03_*.py  - Kappa visualizations               │
└──────────────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────┐
│  4. FULL PIPELINE (2 hours)                                 │
│     eval_demo_gpt5.py       - Production-quality demo       │
│     verify_evaluator.py     - Full verification             │
│     plot_runs.py            - Visualization                 │
└─────────────────────────────────────────────────────────────┘
```
