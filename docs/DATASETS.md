# Dataset Acquisition Plan

This project now supports all target dataset sources in one of two ways:

- **Automatic**: Hugging Face dataset or local manifest that the pipeline can load directly.
- **Manual**: data requires external terms acceptance, account approval, or manual download, then a CSV manifest.

## Automatic Sources Added

### FLEURS Hindi

- Configured as `fleurs_hi`
- Hugging Face path: `google/fleurs`
- Config: `hi_in`
- Current role: train/dev/test fallback and clean benchmark evaluation
- Already loadable in this environment.

### Common Voice Hindi Mirror

- Configured as `common_voice_hi_mirror`
- Hugging Face path: `fsicoli/common_voice_22_0`
- Config: `hi`
- Current role: train/dev/test
- This is an unofficial mirror. The official Mozilla/Hugging Face entry now redirects users to Mozilla Data Collective, so this mirror is the fastest automatic path.

## Manual Sources

### Official Mozilla Common Voice Hindi

Official Common Voice is now distributed through Mozilla Data Collective.

1. Go to `https://commonvoice.mozilla.org/` or Mozilla Data Collective.
2. Create/log in to your account.
3. Accept the dataset terms.
4. Download Hindi scripted speech.
5. Extract it under:

```text
data/raw/common_voice_official_hi/
```

6. Create a manifest CSV:

```text
data/raw/common_voice_official_hi/manifest.csv
```

Columns:

```csv
audio_path,transcript,speaker_id,split,domain,language_mix
clips/common_voice_hi_001.mp3,यह एक उदाहरण है,client001,train,read,hindi
```

If the official folder has the standard Common Voice structure (`hi/train.tsv`, `hi/dev.tsv`, `hi/test.tsv`, `hi/clips/`), generate the manifest automatically:

```powershell
.\.venv\Scripts\python.exe scripts\make_common_voice_manifest.py --root data/raw/common_voice_official_hi/hi
```

This writes:

```text
data/raw/common_voice_official_hi/manifests/common_voice_official_hi.csv
```

### Shrutilipi Hindi

Hugging Face path:

```text
ai4bharat/Shrutilipi
```

Config:

```text
hindi
```

Current status in this environment: requires access approval. To add it:

1. Visit `https://huggingface.co/datasets/ai4bharat/Shrutilipi`.
2. Request/accept access with the same Hugging Face account used by `hf auth login`.
3. After access is granted, set `enabled: true` for `shrutilipi_hi` in `configs/default.yaml`.
4. Run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs/default.yaml
```

### LAHAJA

Hugging Face path:

```text
ai4bharat/Lahaja
```

Config:

```text
default
```

Current status in this environment: requires access approval. To add it:

1. Visit `https://huggingface.co/datasets/ai4bharat/Lahaja`.
2. Request/accept access.
3. After access is granted, set `enabled: true` for `lahaja` in `configs/default.yaml`.
4. Keep it primarily as an accent-robust evaluation set.

### Gram Vaani / OpenSLR 118

Source:

```text
https://www.openslr.org/118/
```

Use for conversational/telephone Hindi. After download and extraction:

```text
data/raw/gram_vaani_slr118/
```

Create one or more CSV manifests under that folder:

```csv
audio_path,transcript,speaker_id,split,domain,language_mix
audio/utt001.wav,मुझे अपना खाता चेक करना है,caller001,train,telephone,hindi
```

Commercial usage may require permission. Confirm license terms before using it outside research/academic experiments.

### SpringLab / SpringX Hindi

Source:

```text
https://asr.iitm.ac.in/
```

After download/extraction:

```text
data/raw/springx_hindi/
```

Create CSV manifests:

```csv
audio_path,transcript,speaker_id,split,domain,language_mix
audio/spx001.wav,आज मौसम बहुत अच्छा है,spk001,train,conversation,hindi
```

## Local Manifest Rules

Every manually added dataset should use:

```csv
audio_path,transcript,speaker_id,split,domain,language_mix
```

Required:

- `audio_path`
- `transcript`
- `speaker_id`
- `split`

Recommended split:

- `train`: 80-90%
- `dev`: 5-10%
- `test`: 5-10%

Do not place the same speaker in both train and dev/test.

## Check Availability

Run:

```powershell
.\.venv\Scripts\python.exe scripts\check_datasets.py --config configs/default.yaml
```

Then prepare data:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs/default.yaml
```
