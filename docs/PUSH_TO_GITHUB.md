# Push This Project to GitHub

This repository should contain code, configs, docs, tests, and lightweight reports.
It should not contain raw datasets, processed datasets, checkpoints, virtual environments,
API keys, Hugging Face caches, or generated training outputs.

## 1. Check What Will Be Tracked

```bash
git status
```

If this folder is not yet a Git repository, initialize it first:

```bash
git init
git branch -M main
```

## 2. Add Files

```bash
git add .
git status
```

Before committing, make sure you do not see files from:

- `data/raw/`
- `data/processed/`
- `outputs/checkpoints/`
- `.venv/`
- `venv/`
- `.env`
- large audio files like `.mp3`, `.wav`, `.flac`
- model files like `.pt`, `.safetensors`, `.bin`

## 3. Commit

```bash
git commit -m "Add Hindi ASR fine-tuning pipeline"
```

## 4. Create an Empty GitHub Repo

Create a new empty repository on GitHub. Do not add a README, license, or `.gitignore`
there, because this project already has those files locally.

## 5. Connect Local Repo to GitHub

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME`:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

If `origin` already exists:

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## 6. What Your Friend Must Add Manually

The GitHub repo will not include the actual Common Voice audio or any local custom data.
Your friend should copy/download datasets into the same expected folders, then run:

```bash
python scripts/make_common_voice_manifest.py --root data/raw/common_voice_official_hi/hi
python scripts/check_datasets.py --config configs/serious_gpu_12h.yaml
```

## 7. Secrets

Never commit real API keys.

Use `.env.example` only as a template. For actual runs, set keys in the terminal:

```bash
export OPENAI_API_KEY="your-key"
export HF_TOKEN="your-token"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:HF_TOKEN="your-token"
```
