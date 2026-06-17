# Experiment Specification: Personalization-Placement Ablation

## Research Question
Where should persona/context be injected in a search-agent pipeline: query fan-out, final synthesis, both, or mixed fan-out?

## Variants
- **V0_generic_single**: Raw query only, no persona in fan-out or synthesis.
- **V1_generic_fanout**: Generic fan-out, no persona in synthesis.
- **V2_synthesis_only_personalization**: Generic fan-out, persona in synthesis only.
- **V3_personalized_fanout**: Persona-aware fan-out plus persona-aware synthesis.
- **V4_mixed_fanout**: Mixed fan-out plus persona-aware synthesis.

## Task Types
- `search_native`: e.g., shopping/product comparison, textbook recommendation.
- `synthesis_native`: e.g., technical concept explanation, career strategy.

**Note:** `task_type` is purely metadata for analysis. It does NOT alter the generation logic or prompt routing at runtime. We want to observe whether the same pipeline configuration naturally behaves differently depending on the inherent nature of the task.

## Expected Hypotheses
- Personalized fan-out (`V3`) should provide a larger marginal benefit on **search-native tasks** where specific retrieved evidence is critical.
- Personalized synthesis (`V2`) should provide a larger marginal benefit on **synthesis-native tasks** where the main challenge is tone/explanation-level alignment rather than new fact retrieval.

## Main Comparisons
- **V2 - V1**: The marginal effect of persona-aware synthesis.
- **V3 - V2**: The marginal effect of adding persona-aware fan-out.
- **V3 - V1**: The full effect of pipeline personalization.
- **V4 - V3**: The effect of mixing generic queries into a personalized fan-out.
