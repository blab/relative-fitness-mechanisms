python3 ../../scripts/prepare-data.py \
    --seq-counts global.tsv.gz \
    --max-date "2024-06-01" \
    --included-days 460 \
    --location-min-seq 300 \
    --location-min-seq-days 60 \
    --output-seq-counts prepared-sequence-counts.tsv

python3 ../../scripts/collapse-counts.py \
    --seq-counts prepared-sequence-counts.tsv \
    --collapse-threshold 5000 \
    --output-seq-counts collapsed-sequence-counts-global.tsv
