# Experiments

Use this folder for run notes, copied configs, command logs, and hardware observations.

Recommended first pass:

1. Prepare accessible datasets.
2. Run `whisper_baseline` on the test split.
3. Fine-tune `openai/whisper-small` with LoRA for 3 epochs.
4. Evaluate the best checkpoint in both normalization modes.
5. Apply constrained LLM correction only to validation/test prediction files.
6. Generate `reports/experiment_report.md`.

