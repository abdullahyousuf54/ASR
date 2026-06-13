# Serious Hindi ASR GPU Run

This guide is for running the proposed Whisper Base + LoRA Hindi ASR experiment on a friend PC or Colab-style GPU machine.

Use this config:

```text
configs/serious_gpu_12h.yaml
```

It trains:

- `openai/whisper-base`
- LoRA adapters on `q_proj` and `v_proj`
- Indic-normalized Hindi labels
- Official local Common Voice Hindi, FLEURS Hindi, and optional custom Hindi/Hinglish manifests

It does **not** start automatically. Run the commands below manually.

## 1. Expected Data Layout

Official Common Voice Hindi should be copied here:

```text
data/raw/common_voice_official_hi/hi/
```

Expected files:

```text
data/raw/common_voice_official_hi/hi/clips/
data/raw/common_voice_official_hi/hi/train.tsv
data/raw/common_voice_official_hi/hi/dev.tsv
data/raw/common_voice_official_hi/hi/test.tsv
data/raw/common_voice_official_hi/hi/clip_durations.tsv
```

Optional custom Hindi/Hinglish manifests:

```text
data/raw/custom_asr/manifests/*.csv
```

CSV format:

```csv
audio_path,transcript,speaker_id,split,domain,language_mix
audio/audio001.wav,आज meeting कितने बजे है,spk001,train,office,hinglish
```

## 2. Windows GPU Setup

Create and activate the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Install CUDA PyTorch explicitly. Pick the CUDA wheel that matches the machine. For recent NVIDIA drivers, this worked on our machine:

```powershell
python -m pip install --upgrade --force-reinstall torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --upgrade --force-reinstall "fsspec[http]>=2023.1.0,<=2024.6.1"
```

Check CUDA:

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected:

```text
torch.cuda.is_available() = True
```

## 3. Colab Setup

In Colab, upload or mount the project folder, then run:

```bash
pip install -r requirements.txt
pip install --upgrade --force-reinstall "fsspec[http]>=2023.1.0,<=2024.6.1"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Colab usually already has CUDA PyTorch. If CUDA is false, switch runtime to GPU.

## 4. Build Official Common Voice Manifest

Run:

```powershell
.\.venv\Scripts\python.exe scripts\make_common_voice_manifest.py --root data/raw/common_voice_official_hi/hi
```

Expected output:

```text
data/raw/common_voice_official_hi/manifests/common_voice_official_hi.csv
```

On our local copy this produced:

```text
train: 4894 clips, 5.86 hours
dev:   2809 clips, 3.71 hours
test:  3326 clips, 4.73 hours
```

## 5. Check Dataset Availability

```powershell
.\.venv\Scripts\python.exe scripts\check_datasets.py --config configs/serious_gpu_12h.yaml
```

At minimum, these should be ready:

```text
common_voice_hi_local: ready
fleurs_hi: ready
```

Custom data is optional.

## 6. Prepare Data

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs/serious_gpu_12h.yaml
```

This writes:

```text
data/processed/hindi_asr_serious_gpu_12h
```

## 7. Run Baseline

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline.py --config configs/serious_gpu_12h.yaml --run-name whisper_base_serious_baseline
```

Outputs:

```text
outputs/predictions/whisper_base_serious_baseline_test.csv
outputs/metrics/whisper_base_serious_baseline_test.json
```

## 8. Train Whisper Base LoRA

```powershell
.\.venv\Scripts\python.exe scripts\train_whisper.py --config configs/serious_gpu_12h.yaml
```

Main checkpoint directory:

```text
outputs/checkpoints/whisper-base-hi-serious-lora
```

This config is designed as a practical 8-12 hour class run depending on GPU speed.

## 9. Evaluate Fine-Tuned Model

Evaluate the final adapter:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_model.py --config configs/serious_gpu_12h.yaml --checkpoint outputs/checkpoints/whisper-base-hi-serious-lora --run-name whisper_base_serious_lora
```

If training logs show a best checkpoint like `checkpoint-500`, evaluate that too:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_model.py --config configs/serious_gpu_12h.yaml --checkpoint outputs/checkpoints/whisper-base-hi-serious-lora\checkpoint-500 --run-name whisper_base_serious_lora_best
```

## 10. Optional LLM Correction

LLM correction is post-processing, not training.

First edit:

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

Set the API key:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
```

Run LLM correction on baseline predictions:

```powershell
.\.venv\Scripts\python.exe scripts\llm_post_correct.py --config configs/serious_gpu_12h_llm_openai_compatible.yaml --predictions outputs/predictions/whisper_base_serious_baseline_test.csv
```

Run LLM correction on fine-tuned predictions:

```powershell
.\.venv\Scripts\python.exe scripts\llm_post_correct.py --config configs/serious_gpu_12h_llm_openai_compatible.yaml --predictions outputs/predictions/whisper_base_serious_lora_test.csv
```

Only keep LLM correction if WER/CER improves and manual review shows no meaning drift.

## 11. Generate Reports

General report:

```powershell
.\.venv\Scripts\python.exe scripts\make_report.py --config configs/serious_gpu_12h.yaml
```

Team visual report:

```powershell
.\.venv\Scripts\python.exe scripts\make_team_report.py --baseline-metrics outputs/metrics/whisper_base_serious_baseline_test.json --fine-tuned-metrics outputs/metrics/whisper_base_serious_lora_test.json --baseline-label "Whisper Base Baseline" --fine-tuned-label "Whisper Base LoRA"
```

Outputs:

```text
reports/team_results_report.html
reports/team_results_report.md
reports/team_results_summary.csv
reports/figures/
```

## 12. Recommended Decision Rule

Use the model with the lowest **Indic WER** on the held-out test set.

If LoRA does not beat baseline:

```text
Use Whisper Base baseline + Indic normalization.
Do not deploy the LoRA checkpoint.
Try more Hinglish/in-domain data or lower learning rate.
```

