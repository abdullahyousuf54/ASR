# Hindi / Hinglish Whisper ASR Fine-Tuning Pipeline

This repository contains a reproducible pipeline for fine-tuning Whisper on Hindi ASR data, evaluating Hindi and Hinglish robustness, and optionally applying constrained LLM post-correction.

## Implementation Plan

1. Scaffold a clean Python project with configurable paths, model settings, dataset definitions, training options, and evaluation modes.
2. Prepare datasets from Hugging Face where possible, with graceful skips for gated or manually licensed datasets.
3. Store both transcript forms for every sample:
   - `transcript_raw`: original dataset text after light safety cleanup
   - `transcript_indic`: Indic-normalized Hindi label used for training
4. Train and evaluate these systems:
   - pretrained Whisper baseline
   - fine-tuned Whisper
   - fine-tuned Whisper under both Whisper-style and Indic-normalized evaluation
   - fine-tuned Whisper plus optional LLM post-correction
   - optional Indic-native baseline when compute permits
5. Produce metrics, prediction files, aggregate tables, and a markdown experiment report.

## Project Layout

```text
configs/             YAML experiment configs
data/                Raw and processed datasets
experiments/         Experiment metadata and run notes
outputs/             Checkpoints, predictions, metrics, summaries
reports/             Generated markdown reports
scripts/             CLI entrypoints
src/asr_pipeline/    Reusable pipeline code
tests/               Lightweight unit tests
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For Hugging Face gated datasets, log in first:

```bash
huggingface-cli login
```

## Dataset Notes

The default config prioritizes openly usable data and skips unavailable datasets without failing the whole run.

| Dataset | Default role | Access note |
|---|---:|---|
| Common Voice Hindi | train/dev/test | Requires Mozilla terms acceptance depending on version |
| Shrutilipi Hindi | train/dev | CC BY 4.0 on Hugging Face |
| Gram Vaani SLR118 | train/dev/eval | Academic use allowed; commercial use requires permission; use local manifests |
| SpringLab / SpringX-Hindi | optional train/dev/test | Use local manifests if available |
| FLEURS Hindi | test | Clean evaluation benchmark |
| LAHAJA | test | CC BY 4.0 accent evaluation |

For local or manually downloaded datasets, create CSV or JSONL manifests with:

```text
audio_path,transcript,speaker_id,split
```

## Quick Start

Prepare accessible datasets:

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

Run pretrained Whisper baseline:

```bash
python scripts/run_baseline.py --config configs/default.yaml
```

Fine-tune Whisper with the default LoRA setup:

```bash
python scripts/train_whisper.py --config configs/default.yaml
```

Evaluate a fine-tuned checkpoint:

```bash
python scripts/evaluate_model.py --config configs/default.yaml --checkpoint outputs/checkpoints/whisper-small-hi
```

Run optional LLM post-correction on prediction files:

```bash
python scripts/llm_post_correct.py --config configs/default.yaml --predictions outputs/predictions/fine_tuned_test.csv
```

Build the final report and summary tables:

```bash
python scripts/make_report.py --config configs/default.yaml
```

## Normalization Strategy

Training labels use Indic normalization to preserve Hindi script fidelity. Evaluation always reports two modes:

| Mode | Purpose |
|---|---|
| `whisper` | Comparable to Whisper-style aggressive normalization; can produce lower measured WER |
| `indic` | More faithful for Hindi diacritics, nukta, matras, and complex characters |

The report keeps both raw and normalized text. Do not silently choose one metric.

## LLM Post-Correction

LLM correction is optional and constrained. The correction prompt requires:

- no paraphrasing
- no added information
- no reordering unless necessary
- preserve language, names, and code-mixed words
- no-op fallback when confidence is low

Configure `llm.provider` as `none`, `local_http`, or `openai_compatible`.

## Expected Outputs

```text
outputs/checkpoints/              trained models/adapters
outputs/predictions/              CSV/JSONL per-sample hypotheses
outputs/metrics/                  JSON metric files
outputs/summaries/summary.csv     WER/CER comparison table
reports/experiment_report.md      final experiment report
```

At completion, `scripts/make_report.py` prints:

1. best-performing model/configuration
2. WER/CER before and after fine-tuning
3. whether LLM correction helped
4. a deployment recommendation for Hindi/Hinglish ASR

