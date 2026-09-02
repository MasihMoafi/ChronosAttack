# ChronosAttack: Adversarial Tool Scheduling Attacks on LLM Agents

ChronosAttack studies whether an attacker can change the decision of a tool-using large language model (LLM) agent **without changing the contents of any tool response**.

Instead of modifying, injecting, deleting, or fabricating tool outputs, ChronosAttack changes only **when authentic tool responses are delivered**. Small non-negative delays can change the order in which the same evidence reaches the agent, which can change the agent's final decision.

The project evaluates this behavior under stateful and stateless processing, different scheduling perturbation sizes, and two defenses: barrier synchronization and order consistency.

The paper evaluates four LLM families: GPT-5.6 Sol, Gemini 3.6 Flash, DeepSeek V4 Flash, and Claude Sonnet 4.6.

## Overview

![ChronosAttack study overview](chronosattack_study_figure.drawio.png)

The experiment keeps tool payloads authentic and unchanged. Only response timing and the resulting observation order are varied.

For the two paper scenarios, the agent receives three tool observations and chooses one of three candidates. The main evaluation includes:

1. Natural versus attacked schedules under stateful processing.
2. Stateful versus stateless processing.
3. A simultaneous control where all evidence is provided together.
4. Schedule perturbations with 0, 1, or 2 pairwise inversions.
5. Barrier synchronization.
6. Order-consistency evaluation across all six permutations of the three observations.

## Project Structure

The source and curated extension record contain:

```text
ChronosAttack/
├── README.md
├── RESULTS.md
├── requirements.txt
├── helper.py
├── chronos_multimodel_full.py
├── chronos_provider_adapter.py
├── run_experiments_tmux.sh
├── test_chronos_extension.py
├── chronos_luna_paper_20260901/
│   ├── run_manifest.json
│   └── codex/*_trials.json
└── chronosattack_study_figure.drawio.png
```

`helper.py` contains the scenarios, authentic tool payloads, scheduling logic, prompts, stateful execution, integrity checks, and shared utilities.

`chronos_multimodel_full.py` is the main experiment runner. It executes the confirmation, perturbation-budget, barrier-defense, and order-consistency experiments.

`chronos_provider_adapter.py` provides the provider interface used to run the same experiment logic across the evaluated model families.

The repository intentionally does not include old exploratory results, checkpoints, API ledgers, or intermediate experiment versions.

## Installation

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## API Credentials

The multi-model experiment requires valid credentials for the providers you want to evaluate.

Configure the required API credentials in your environment before running the experiment. Do not place API keys directly in the source code or commit them to a public repository.

In PowerShell:

```powershell
$env:OPENAI_API_KEY="your-openai-key"
$env:GEMINI_API_KEY="your-gemini-key"
$env:DEEPSEEK_API_KEY="your-deepseek-key"
$env:ANTHROPIC_API_KEY="your-anthropic-key"
```

Set the corresponding credentials required by `chronos_provider_adapter.py` for any other providers you plan to run.

### Codex subscription

The `codex` provider uses an existing ChatGPT/Codex CLI subscription instead
of an API key. Authenticate once with `codex login`, then select it explicitly:

```bash
python chronos_multimodel_full.py \
    --providers codex \
    --codex-model gpt-5.6-terra \
    --out chronos_codex_results
```

Codex results are labelled `codex_cli_agent`, not as raw provider-API calls.
Each call runs in an isolated, read-only temporary workspace, ignores local
Codex configuration and repository instructions, enforces the requested JSON
schema, and rejects results that used unrelated shell, web, MCP, filesystem,
or subagent tools. The ChronosAttack tool observations and scheduling logic
remain unchanged.

## Running the Experiments

The main runner supports four experiment groups:

- `confirmation`
- `budget`
- `barrier`
- `consistency`

### Preflight check

```powershell
python .\chronos_multimodel_full.py --preflight-only
```

### Run the complete evaluation

```powershell
python .\chronos_multimodel_full.py `
    --out .\chronos_multimodel_results
```

By default, the runner evaluates the configured OpenAI, Gemini, DeepSeek, and Claude models and executes all four experiment groups.

The default paper settings are:

- 30 stateful trials per natural or attacked schedule
- 10 stateless trials per serialization
- 10 simultaneous controls
- 15 trials per schedule-perturbation level
- 15 barrier-defense trials
- 10 order-consistency trials, each evaluating all six permutations

### Run selected providers

```powershell
python .\chronos_multimodel_full.py `
    --providers openai claude `
    --out .\chronos_selected_models
```

### Run selected experiments

```powershell
python .\chronos_multimodel_full.py `
    --experiments confirmation budget `
    --out .\chronos_confirmation_budget
```

### Resume an interrupted run

```powershell
python .\chronos_multimodel_full.py `
    --out .\chronos_multimodel_results `
    --resume
```

### Run Codex in a background tmux session

```bash
./run_experiments_tmux.sh chronos_codex_results
```

The script runs all four experiment groups with the paper defaults and resumes
the same output directory after an interruption.

## Output Files

Experiment outputs are written to the directory specified with `--out`. Each provider receives its own output directory containing trial-level results, checkpoints, and audit information.

These generated files are not required in the source repository and should normally be excluded from version control.

## Reproducibility Notes

ChronosAttack uses fixed paper scenarios, fixed authentic tool payloads, fixed natural and attacked schedules, SHA-256 payload integrity checks, and fixed random seeds for experiment ordering.

The scheduling implementation only permits non-negative delays. It does not accelerate observations, modify payloads, add observations, remove observations, or duplicate observations.

The same scenario definitions, prompts, payloads, and schedule logic are reused across model providers.

## Author

**Arash Vashagh**  
Faculty of Computer Science  
University of New Brunswick  
Fredericton, New Brunswick, Canada

Paper: **ChronosAttack: Adversarial Tool Scheduling Attacks on LLM Agents**
