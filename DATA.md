# Data availability notes

This repository stores the analysis code and the processed source-data package supporting the main-text figures.

## Included data

- `source_data/`: figure-level source data for main-text Figs. 1-8.
- `Source_Data.zip`: compressed copy of the same source-data folder.
- `source_data/SOURCE_DATA_MANIFEST.csv`: file inventory with sizes and SHA-256 checksums.

## Not included

The workflow expects local geospatial source data, extracted predictor tables, trained GWR artifacts and generated prediction or optimization outputs under `data/` and `outputs/` when rerunning the full pipeline. Large third-party raw predictor datasets are not redistributed here and should be obtained from the original data providers cited in the manuscript.
