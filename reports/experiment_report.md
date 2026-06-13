# Hindi / Hinglish ASR Experiment Report

## Datasets Used

- `common_voice_hi`: kind=hf, train=True, eval=True. 
- `shrutilipi_hi`: kind=hf, train=True, eval=False. 
- `fleurs_hi`: kind=hf, train=True, eval=True. Used for training fallback only because Common Voice was unavailable and Shrutilipi/LAHAJA were gated for this account.
- `lahaja`: kind=hf, train=False, eval=True. 
- `gram_vaani_slr118`: kind=local_manifest, train=True, eval=True. Academic use allowed; commercial use requires permission.
- `springx_hindi`: kind=local_manifest, train=False, eval=True. 

## Preprocessing

- Audio is decoded as mono 16 kHz and filtered by configured duration limits.
- Raw transcript text is retained as `transcript_raw`.
- Training labels use `transcript_indic` after Hindi-preserving Indic normalization.
- Empty, corrupt, missing, and over-long samples are skipped.

## Normalization Strategy

Both Whisper-style and Indic-normalized metrics are reported. Whisper-style normalization can make WER look lower, while Indic normalization is more faithful for Hindi script details.

## Summary Metrics

| run_name                  | split   | dataset   | mode    | stage   |     wer |     cer |   exact_match |      rtf |   num_samples | metrics_file                                        |
|:--------------------------|:--------|:----------|:--------|:--------|--------:|--------:|--------------:|---------:|--------------:|:----------------------------------------------------|
| fine_tuned_gpu_final      | test    | overall   | whisper | asr     | 1.24602 | 1.48025 |             0 | 0.225521 |            64 | outputs\metrics\fine_tuned_gpu_final_test.json      |
| fine_tuned_gpu_final      | test    | fleurs_hi | whisper | asr     | 1.24602 | 1.48025 |             0 | 0.225521 |            64 | outputs\metrics\fine_tuned_gpu_final_test.json      |
| fine_tuned_gpu_final      | test    | overall   | indic   | asr     | 1.32018 | 1.28048 |             0 | 0.225521 |            64 | outputs\metrics\fine_tuned_gpu_final_test.json      |
| fine_tuned_gpu_final      | test    | fleurs_hi | indic   | asr     | 1.32018 | 1.28048 |             0 | 0.225521 |            64 | outputs\metrics\fine_tuned_gpu_final_test.json      |
| fine_tuned_gpu            | test    | overall   | whisper | asr     | 1.24602 | 1.48025 |             0 | 0.261116 |            64 | outputs\metrics\fine_tuned_gpu_test.json            |
| fine_tuned_gpu            | test    | fleurs_hi | whisper | asr     | 1.24602 | 1.48025 |             0 | 0.261116 |            64 | outputs\metrics\fine_tuned_gpu_test.json            |
| fine_tuned_gpu            | test    | overall   | indic   | asr     | 1.32018 | 1.28048 |             0 | 0.261116 |            64 | outputs\metrics\fine_tuned_gpu_test.json            |
| fine_tuned_gpu            | test    | fleurs_hi | indic   | asr     | 1.32018 | 1.28048 |             0 | 0.261116 |            64 | outputs\metrics\fine_tuned_gpu_test.json            |
| fine_tuned                | test    | overall   | whisper | asr     | 1.11928 | 1.14653 |             0 | 0.220243 |            25 | outputs\metrics\fine_tuned_test.json                |
| fine_tuned                | test    | fleurs_hi | whisper | asr     | 1.11928 | 1.14653 |             0 | 0.220243 |            25 | outputs\metrics\fine_tuned_test.json                |
| fine_tuned                | test    | overall   | indic   | asr     | 1.07715 | 1.02646 |             0 | 0.220243 |            25 | outputs\metrics\fine_tuned_test.json                |
| fine_tuned                | test    | fleurs_hi | indic   | asr     | 1.07715 | 1.02646 |             0 | 0.220243 |            25 | outputs\metrics\fine_tuned_test.json                |
| whisper_base_baseline_gpu | test    | overall   | whisper | asr     | 1.07293 | 1.12215 |             0 | 0.185357 |            64 | outputs\metrics\whisper_base_baseline_gpu_test.json |
| whisper_base_baseline_gpu | test    | fleurs_hi | whisper | asr     | 1.07293 | 1.12215 |             0 | 0.185357 |            64 | outputs\metrics\whisper_base_baseline_gpu_test.json |
| whisper_base_baseline_gpu | test    | overall   | indic   | asr     | 1.07644 | 1.00296 |             0 | 0.185357 |            64 | outputs\metrics\whisper_base_baseline_gpu_test.json |
| whisper_base_baseline_gpu | test    | fleurs_hi | indic   | asr     | 1.07644 | 1.00296 |             0 | 0.185357 |            64 | outputs\metrics\whisper_base_baseline_gpu_test.json |
| whisper_baseline          | test    | overall   | whisper | asr     | 1.11928 | 1.14691 |             0 | 0.203934 |            25 | outputs\metrics\whisper_baseline_test.json          |
| whisper_baseline          | test    | fleurs_hi | whisper | asr     | 1.11928 | 1.14691 |             0 | 0.203934 |            25 | outputs\metrics\whisper_baseline_test.json          |
| whisper_baseline          | test    | overall   | indic   | asr     | 1.07715 | 1.02675 |             0 | 0.203934 |            25 | outputs\metrics\whisper_baseline_test.json          |
| whisper_baseline          | test    | fleurs_hi | indic   | asr     | 1.07715 | 1.02675 |             0 | 0.203934 |            25 | outputs\metrics\whisper_baseline_test.json          |

## Best Configuration

Best Indic WER currently comes from `whisper_base_baseline_gpu` with WER=1.0764 and CER=1.0030.

## Recommendations

For production Hindi/Hinglish ASR, start with the fine-tuned Whisper checkpoint selected by lowest Indic WER, keep Whisper-style metrics only for comparability, and enable LLM correction only if it improves WER/CER on code-mixed and number-heavy validation subsets without raising semantic drift during manual review.
