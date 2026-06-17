# Notebook Path Update

Because of the repository refactor for `placement_ablation_v1`, the notebook `analysis.ipynb` should be updated to read from the new config-driven CSV paths.

Please update any internal pandas `read_csv` calls to point to:
- `../outputs/placement_ablation_v1/summary_by_variant.csv`
- `../outputs/placement_ablation_v1/summary_by_variant_task_type.csv`
- `../outputs/placement_ablation_v1/summary_by_variant_task_category.csv`

The new format includes the columns:
- `variant`
- `task_type` (if applicable)
- `task_category` (if applicable)
- Mean scores for both final response (`final_intent_satisfaction_mean`, etc.) and fan-out queries (`fanout_persona_field_use_mean`, etc.).
