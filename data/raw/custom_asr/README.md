# Custom Hindi/Hinglish ASR Data

Fastest format:

```csv
audio_path,transcript,speaker_id,split,domain,language_mix
audio/audio001.wav,आज meeting कितने बजे है,spk001,train,office,hinglish
audio/audio002.wav,मेरा नाम राहुल है,spk002,dev,general,hindi
audio/audio003.wav,payment failed दिखा रहा है,spk003,test,support,hinglish
```

Required columns:

- `audio_path`
- `transcript`
- `speaker_id`
- `split`

Put audio files under `data/raw/custom_asr/audio/` and CSV manifests under `data/raw/custom_asr/manifests/`.

Allowed audio formats depend on installed decoders, but `.wav`, `.flac`, and `.mp3` are usually fine. The pipeline converts loaded audio to mono 16 kHz.
