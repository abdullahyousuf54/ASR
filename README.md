# Hindi/Hinglish Whisper ASR Fine-Tuning Pipeline

This project trains and evaluates a Whisper-based Hindi/Hinglish ASR model using a practical 12-hour GPU plan.

The main run uses:

- Official Common Voice Hindi downloaded locally
- FLEURS Hindi from Hugging Face
- Optional local custom Hindi/Hinglish audio
- Whisper Base + LoRA fine-tuning
- Indic normalization for training labels
- Whisper-style and Indic-style evaluation
- Optional OpenAI-compatible LLM post-correction

This README is the handoff guide for running the project on another GPU machine.

## 1. What Should Be in GitHub

GitHub should contain only code, configs, docs, tests, templates, and lightweight reports.

Do not commit:

- `data/raw/common_voice_official_hi/hi/clips/`
- `data/processed/`
- `outputs/checkpoints/`
- `.venv/`
- `venv/`
- `.env`
- `.mp3`, `.wav`, `.flac`
- `.pt`, `.safetensors`, `.bin`

The `.gitignore` already protects these, but still check `git status` before pushing.

## 2. Recommended Hardware

Minimum practical GPU:

- 8 GB VRAM for a smoother run
- 4 GB VRAM may work with very small batches, but training will be slower and less stable

Recommended for the 12-hour plan:

- NVIDIA GPU with CUDA
- 12 GB+ VRAM if available
- 20 GB+ free disk space
- Python 3.10 or 3.11

The serious config is:

```text
configs/serious_gpu_12h.yaml
```

It uses:

```text
openai/whisper-base
LoRA rank 8
max_train_samples: 6000
max_dev_samples: 500
max_eval_samples: 500
epochs: 3
```

## 3. Create Python Environment

From the project root:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\activate
```

On Linux / Colab:

```bash
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

For NVIDIA CUDA, install GPU PyTorch after requirements if needed:

```bash
pip install --upgrade --force-reinstall torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

Then verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected:

```text
True
```

for `torch.cuda.is_available()`.

## 4. Hugging Face Login

For this 12-hour plan, Hugging Face is used for:

- downloading `openai/whisper-base`
- downloading `google/fleurs`

These usually work without login, but logging in avoids many download/rate-limit problems.

Use:

```bash
hf auth login
```

Do not use:

```bash
huggingface-cli login
```

because it is deprecated in newer Hugging Face versions.

Create a Hugging Face token:

```text
Hugging Face profile -> Settings -> Access Tokens -> New token -> Read
```

Paste that token when `hf auth login` asks for it.

Do not commit this token to GitHub.

Optional environment variable:

Windows PowerShell:

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

Linux / Colab:

```bash
export HF_TOKEN="hf_your_token_here"
```

## 5. Add Official Common Voice Hindi

This project expects the official Common Voice Hindi files locally.

Download Hindi from:

```text
https://commonvoice.mozilla.org/
```

After extraction, the folder must look like this:

```text
data/raw/common_voice_official_hi/hi/
  clips/
    common_voice_hi_....mp3
  train.tsv
  dev.tsv
  test.tsv
  validated.tsv
  invalidated.tsv
  other.tsv
  reported.tsv
  clip_durations.tsv
```

The important files are:

- `clips/`
- `train.tsv`
- `dev.tsv`
- `test.tsv`

If your extracted folder is named something else, move or rename it so the final path is exactly:

```text
data/raw/common_voice_official_hi/hi
```

Then create the local manifest:

```bash
python scripts/make_common_voice_manifest.py --root data/raw/common_voice_official_hi/hi
```

This creates:

```text
data/raw/common_voice_official_hi/manifests/common_voice_official_hi.csv
```

That manifest is what the training config reads.

## 6. Hugging Face Dataset Used in This Plan

The 12-hour config uses only this Hugging Face dataset:

```text
google/fleurs
config: hi_in
```

You do not need to manually download it.

This command downloads and prepares it automatically:

```bash
python scripts/prepare_data.py --config configs/serious_gpu_12h.yaml
```

This 12-hour plan does not use Shrutilipi or LAHAJA. Do not spend time trying to add them for this run.

## 7. Optional Custom Hindi/Hinglish Data

If you have extra local Hindi or Hinglish audio, put it here:

```text
data/raw/custom_asr/
```

Recommended layout:

```text
data/raw/custom_asr/
  audio/
    sample001.wav
    sample002.wav
  manifests/
    custom_train.csv
```

CSV format:

```csv
audio_path,transcript,speaker_id,split,domain,language_mix
data/raw/custom_asr/audio/sample001.wav,आज meeting कितने बजे है,spk001,train,office,hinglish
data/raw/custom_asr/audio/sample002.wav,मेरा नाम राहुल है,spk002,dev,general,hindi
data/raw/custom_asr/audio/sample003.wav,payment failed दिखा रहा है,spk003,test,support,hinglish
```

Required columns:

- `audio_path`
- `transcript`
- `speaker_id`
- `split`

Allowed split values:

- `train`
- `dev`
- `test`

If you do not have custom data, leave this folder empty. The pipeline will continue.

## 8. Check Dataset Availability

Run:

```bash
python scripts/check_datasets.py --config configs/serious_gpu_12h.yaml
```

Expected result:

- `common_voice_hi_local` should be ready after manifest creation
- `fleurs_hi` should be ready if Hugging Face download works
- `custom_hindi_hinglish` is optional

If Common Voice is not ready, check:

```text
data/raw/common_voice_official_hi/hi/train.tsv
data/raw/common_voice_official_hi/hi/dev.tsv
data/raw/common_voice_official_hi/hi/test.tsv
data/raw/common_voice_official_hi/hi/clips/
```

and rerun:

```bash
python scripts/make_common_voice_manifest.py --root data/raw/common_voice_official_hi/hi
```

## 9. Run the Pipeline One Step at a Time

Run these commands in order. Wait for each command to finish before starting the next one.

### Step 1: Prepare Data

```bash
python scripts/prepare_data.py --config configs/serious_gpu_12h.yaml
```

This will:

- load Common Voice from the local manifest
- download/load FLEURS Hindi
- load optional custom Hindi/Hinglish data if present
- validate audio
- create raw transcript and Indic-normalized transcript fields
- save the processed dataset to:

```text
data/processed/hindi_asr_serious_gpu_12h
```

### Step 2: Run Whisper Baseline

```bash
python scripts/run_baseline.py --config configs/serious_gpu_12h.yaml --run-name whisper_base_serious_baseline
```

This evaluates pretrained `openai/whisper-base` before fine-tuning.

Outputs:

```text
outputs/predictions/whisper_base_serious_baseline_test.csv
outputs/metrics/whisper_base_serious_baseline_test_metrics.json
```

### Step 3: Train Whisper Base with LoRA

```bash
python scripts/train_whisper.py --config configs/serious_gpu_12h.yaml
```

This trains LoRA adapter weights, not a full Whisper model from scratch.

Checkpoint output:

```text
outputs/checkpoints/whisper-base-hi-serious-lora
```

Training logs:

```text
runs/
```

If training crashes due to GPU memory, reduce these in `configs/serious_gpu_12h.yaml`:

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_train_samples: 6000
```

Try lowering:

```yaml
max_train_samples: 3000
max_dev_samples: 300
```

### Step 4: Evaluate Fine-Tuned Model

```bash
python scripts/evaluate_model.py --config configs/serious_gpu_12h.yaml --checkpoint outputs/checkpoints/whisper-base-hi-serious-lora --run-name whisper_base_serious_lora
```

Outputs:

```text
outputs/predictions/whisper_base_serious_lora_test.csv
outputs/metrics/whisper_base_serious_lora_test_metrics.json
```

### Step 5: Make Report

```bash
python scripts/make_team_report.py --config configs/serious_gpu_12h.yaml --baseline-metrics outputs/metrics/whisper_base_serious_baseline_test_metrics.json --fine-tuned-metrics outputs/metrics/whisper_base_serious_lora_test_metrics.json --baseline-label "Whisper Base Baseline" --fine-tuned-label "Whisper Base LoRA Fine-Tuned"
```

Report outputs:

```text
reports/team_results_report.md
reports/team_results_report.html
reports/team_results_summary.csv
reports/figures/team_wer_cer_comparison.png
reports/figures/team_error_breakdown_indic.png
reports/figures/team_runtime_rtf.png
```

## 10. Optional OpenAI LLM Correction

LLM correction is optional and runs after ASR evaluation.

It does not train Whisper. It only post-processes the generated ASR text.

Use it to try to improve:

- spelling
- punctuation
- Hindi/English code-mixed words
- numbers
- obvious ASR errors

It must not:

- paraphrase
- add new information
- change meaning
- reorder the sentence unnecessarily

### OpenAI API Key

Do not write the API key directly into YAML.

Set it in the terminal.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
```

Linux / Colab:

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### LLM Config File

Use:

```text
configs/serious_gpu_12h_llm_openai_compatible.yaml
```

Check these lines:

```yaml
llm:
  provider: openai_compatible
  endpoint: https://api.openai.com/v1/chat/completions
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
```

If `endpoint` is empty or `model` says `REPLACE_WITH_MODEL_NAME`, fill them exactly like above.

### Run LLM Correction

```bash
python scripts/llm_post_correct.py --config configs/serious_gpu_12h_llm_openai_compatible.yaml --predictions outputs/predictions/whisper_base_serious_lora_test.csv
```

Then regenerate the report if LLM metrics are produced:

```bash
python scripts/make_team_report.py --config configs/serious_gpu_12h.yaml --baseline-metrics outputs/metrics/whisper_base_serious_baseline_test_metrics.json --fine-tuned-metrics outputs/metrics/whisper_base_serious_lora_test_metrics.json --baseline-label "Whisper Base Baseline" --fine-tuned-label "Whisper Base LoRA Fine-Tuned"
```

## 11. What to Send Back After the Run

After your friend finishes, ask them to send back:

```text
outputs/metrics/
outputs/predictions/
reports/
```

They do not need to send:

```text
data/raw/
data/processed/
outputs/checkpoints/
```

unless you specifically need the trained adapter.

If you need the trained LoRA adapter, send:

```text
outputs/checkpoints/whisper-base-hi-serious-lora/
```

This folder may be large.

## 12. Common Problems

### `hf` command not found

Run:

```bash
pip install -U huggingface_hub
```

Then:

```bash
hf auth login
```

### CUDA is false

Check NVIDIA driver and reinstall CUDA PyTorch:

```bash
pip install --upgrade --force-reinstall torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### Common Voice not found

Make sure this exact folder exists:

```text
data/raw/common_voice_official_hi/hi
```

and contains:

```text
clips/
train.tsv
dev.tsv
test.tsv
```

Then run:

```bash
python scripts/make_common_voice_manifest.py --root data/raw/common_voice_official_hi/hi
```

### Out of memory during training

Edit:

```text
configs/serious_gpu_12h.yaml
```

Lower:

```yaml
training:
  max_train_samples: 3000
  max_dev_samples: 300
```

Keep:

```yaml
per_device_train_batch_size: 1
```

### OpenAI correction fails with authentication error

Set:

```bash
OPENAI_API_KEY
```

in the terminal before running `llm_post_correct.py`.

Do not put the real key in GitHub.

## 13. Final Command Checklist

```bash
python scripts/make_common_voice_manifest.py --root data/raw/common_voice_official_hi/hi
python scripts/check_datasets.py --config configs/serious_gpu_12h.yaml
python scripts/prepare_data.py --config configs/serious_gpu_12h.yaml
python scripts/run_baseline.py --config configs/serious_gpu_12h.yaml --run-name whisper_base_serious_baseline
python scripts/train_whisper.py --config configs/serious_gpu_12h.yaml
python scripts/evaluate_model.py --config configs/serious_gpu_12h.yaml --checkpoint outputs/checkpoints/whisper-base-hi-serious-lora --run-name whisper_base_serious_lora
python scripts/make_team_report.py --config configs/serious_gpu_12h.yaml --baseline-metrics outputs/metrics/whisper_base_serious_baseline_test_metrics.json --fine-tuned-metrics outputs/metrics/whisper_base_serious_lora_test_metrics.json --baseline-label "Whisper Base Baseline" --fine-tuned-label "Whisper Base LoRA Fine-Tuned"
```

Optional LLM correction:

```bash
python scripts/llm_post_correct.py --config configs/serious_gpu_12h_llm_openai_compatible.yaml --predictions outputs/predictions/whisper_base_serious_lora_test.csv
```
