## Provisioning recent counts

We can prune and collapse the pango lineages based on the global sequence counts using the following scripts. This is implemented in the script `./generate-counts.sh`.

### Preparing data for countries of interest

```
python3 ./scripts/prepare-data.py \
    --seq-counts global.tsv.gz \
    --max-date "2024-03-01" \
    --included-days 365 \
    --location-min-seq 300 \
    --location-min-seq-days 30 \
    --output-seq-counts prepared-sequence-counts.tsv
```

### Collapsing small lineages with script

```bash
python3 collapse-counts.py \
    --seq-counts prepared-sequence-counts.tsv \
    --collapse-threshold 5000 \
    --output-seq-counts collapsed-sequence-counts-global.tsv
```

### Output

This will output a final file of sequence counts called `collapsed-sequences-counts-global.tsv`.
