# results/

Lightweight, committed, **permanent** record of the pipeline's output. Not a
dump — only the artifacts we would regret losing. Compare to:
- `runs/` — scratch/ephemeral, gitignored, remote-only. Full training weights
  and raw per-instance predictions live here until they get synced.
- `docs/ref_results/` — gitignored working copies for the HTML reports.

Layout:

```
results/
  or_phase0/
    iso/
      batch_meta.json
      batch_results.csv
      predictions.jsonl.gz       # per-config concat; one file per shard if large
      README.md
    ctxw30/
      ...
  parlasent_zs/
    on_rog_full/
      batch_meta.json
      batch_results.csv
      predictions.jsonl.gz
    on_parlasent/
      ...
  finetune/
    <stamp>/
      matrix_summary.json
      matrix_long.csv
      matrix_report.md
      heatmap_*.png
      cells/<backbone>__<mixture>/
        summary.json
        predictions_<test_source>.jsonl.gz
```

Sync workflow (post-run, remote → local → commit):

```bash
# on remote, pull the OR batch + matrix stamp
rsync -av --include='batch_meta.json' --include='batch_results.csv' \
      --include='*/predictions.jsonl' --include='**/summary.json' \
      --include='**/matrix_summary.json' --include='**/matrix_long.csv' \
      --include='**/matrix_report.md' --include='**/heatmap_*.png' \
      --exclude='*' runs/batch_FULLROG_<stamp>/ results/or_phase0/iso/
```

Or run `scripts/sync_results.py` (see that script for the batched version).
