# LLM Post-Correction

LLM correction is optional post-processing for ASR predictions.

It should be used only after ASR inference:

```text
audio
  -> Whisper transcript
  -> constrained LLM correction
  -> corrected transcript
  -> WER/CER comparison
```

The LLM is not used during LoRA training.

## Rules

The correction prompt enforces:

- do not paraphrase
- do not add information
- do not translate
- preserve Hindi and English language choice
- preserve named entities as much as possible
- return unchanged text if uncertain

## OpenAI-Compatible Provider

Copy or edit:

```text
configs/serious_gpu_12h_llm_openai_compatible.yaml
```

Set:

```yaml
llm:
  provider: openai_compatible
  endpoint:
  model: YOUR_MODEL_NAME
  api_key_env: OPENAI_API_KEY
```

For a local OpenAI-compatible server, set `endpoint`, for example:

```yaml
endpoint: http://localhost:8000/v1
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts\llm_post_correct.py --config configs/serious_gpu_12h_llm_openai_compatible.yaml --predictions outputs/predictions/whisper_base_serious_baseline_test.csv
```

Metrics are written to:

```text
outputs/metrics/
```

Corrected predictions are written to:

```text
outputs/predictions/
```

