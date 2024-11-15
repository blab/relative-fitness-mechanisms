# Provisioning sequence count data

### Downloading publicly available count data

We can retrieve global sequence count data at the level of Pango lineage and Nextstrain clade by running the command

```bash
python3 ../scripts/download-sequence-count-data.py
```

This provisions the local sequence count files
- `nextstrain-clade/global.tsv.gz`
- `nextstrain-clade-usa/usa.tsv.gz`
- `pango-lineage/global.tsv.gz`

Further data preparation instructions are in each of the three directories `nextstrain-clade/`, `nextstrain-clade-usa` and `pango-lineage/`. After running these scripts the resulting sequence counts used in analyses are available as:
- `nextstrain-clade/collapsed-sequence-counts-global.tsv`
- `nextstrain-clade-usa/collapsed-sequence-counts-usa.tsv`
- `pango-lineage/collapsed-sequence-counts-global.tsv`
These files are versioned to this repository.
