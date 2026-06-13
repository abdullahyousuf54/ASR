# Optional Custom Hindi/Hinglish ASR Data

Use this folder only if you have extra local Hindi or Hinglish speech data.

Expected layout:

```text
data/raw/custom_asr/
  audio/
    sample001.wav
    sample002.wav
  manifests/
    custom_manifest.csv
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

The pipeline converts loaded audio to mono 16 kHz during preprocessing.
