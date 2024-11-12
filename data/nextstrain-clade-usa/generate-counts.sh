python3 ../../scripts/prepare-data.py \
    --seq-counts usa.tsv.gz \
    --max-date "2022-12-31" \
    --included-days 730 \
    --location-min-seq 300 \
    --location-min-seq-days 365 \
    --output-seq-counts prepared-sequence-counts.tsv

python3 ../../scripts/collapse-counts.py \
    --seq-counts prepared-sequence-counts.tsv \
    --collapse-threshold 5000 \
    --output-seq-counts collapsed-sequence-counts-usa.tsv
