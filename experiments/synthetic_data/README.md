# Synthetic Data Generation Pipeline

This directory contains the pipeline for generating synthetic datasets to evaluate personalization placement in search-based AI agents.

## Purpose

The goal is to create realistic, fictional user profiles and natural "Level-2 ambiguous queries" for testing whether a search-based AI agent can correctly use relevant user information during query fan-out. This benchmark operates across four domains: e-commerce, health/medical, education, and travel.

## Level-2 Ambiguous Queries

A Level-2 ambiguous query is natural and underspecified. The relevant user preferences and constraints are implicit. The agent must inspect the persona/user history to infer what should be included in query fan-out, because the query itself does not explicitly reveal all constraints.

Example:
- **Clear hidden intent**: "Find a quiet, car-free, budget-friendly weekend trip from Mountain View for someone who likes nature and dislikes nightlife."
- **Level-2 ambiguous query**: "Where should I go this weekend?"

## Query Types

Each generated query belongs to one of three categories:
1. **`personalization_required`**: The query is underspecified and a good answer/search plan requires user context.
2. **`personalization_helpful`**: A generic answer is possible, but user context improves depth, framing, or recommendations.
3. **`overpersonalization_trap`**: User context should mostly not affect the answer, or only affect style lightly. This tests whether the agent overuses persona information.

## Workflow

### 1. Generate Users

```bash
python experiments/synthetic_data/generate_users.py --num_users 20 --model gemini-flash-latest
```
Generates synthetic users with demographics, latent profiles, and observable history.

### 2. Generate Queries

```bash
python experiments/synthetic_data/generate_queries.py --queries_per_user_per_domain 2 --model gemini-flash-latest
```
Generates queries for each user across all domains based on the generated profiles.

### 3. Validate Generated Data

```bash
python experiments/synthetic_data/validate_generated_data.py
```
Checks for basic deterministic validity of the generated data schemas.

## Outputs

All outputs are saved to `experiments/synthetic_data/generated/`:
- `users.jsonl` / `queries.jsonl`: Raw JSON objects.
- `users.csv` / `queries.csv`: Tabular overview.
- `users_preview.md` / `queries_preview.md`: Readable markdown for manual inspection.

Integration files are also generated directly in `experiments/`:
- `sample_personas.generated.jsonl`
- `sample_queries.generated.jsonl`

## Code Organization
- `utils.py`: Contains shared parsing, prompt loading, Gemini rate-limiting, schema loading, and JSONL helpers.
- `domain_schemas.yaml`: The single source of truth for domains and query types.
- Generation scripts (`generate_users.py`, `generate_queries.py`) rely on these utilities and should not hardcode domain/query-type constants.

## Limitations
- Health/medical examples remain synthetic and avoid identifying information, but edge cases should be manually reviewed to prevent the model from practicing diagnosis.
- AI-generated personas may lack deep, nuanced consistency across very long histories; their purpose is strictly to test information retrieval and reasoning in the agent's query fan-out phase.
