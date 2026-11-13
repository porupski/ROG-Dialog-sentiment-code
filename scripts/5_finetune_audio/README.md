# 5_finetune_audio — placeholder

Sibling to `4_finetune_text/`, reserved for the audio-side fine-tune matrix once the
text track lands. Mirror the text layout when the time comes:

```
5_finetune_audio/
  finetune_audio.py
  run_matrix.py
  utils_audio_finetune.py
  config.json
  README.md
```

Candidate backbones to benchmark (wav2vec2 / XLS-R / HuBERT / WavLM / Whisper encoder)
plus the matching `sentiment-on-prosody` recipes should land here, not in `4_finetune_text/`.
