# Nationwide sinkhole risk in China

This repository contains the analysis code used for the nationwide sinkhole susceptibility, attribution, climate-change impact and intervention-optimization workflow.

The repository is a code-only release. Large geospatial inputs, intermediate model artifacts and generated outputs are not included because they exceed normal GitHub size limits and may have separate data-source licenses.

## Repository layout

- `code/1_point_generation/`: point-grid generation and positive/negative sample construction.
- `code/2_feature_extraction/`: extraction and cleaning of geospatial, climatic, hydrogeological and socioeconomic predictors.
- `code/3_gwr_model_train/`: national GWR classification model training and calibration.
- `code/4_attribution/`: correlation, SHAP and LiNGAM attribution workflows.
- `code/5_gwr_model_prediction/`: national and scenario-based GWR prediction and mapping.
- `code/5_1_climate_change/`: counterfactual climate-impact calculations and plotting.
- `code/6_NSGA_II_optimization/`: cell-level NSGA-II intervention optimization and strategy figures.
- `code/mgtwr/`: local GWR/MGTWR helper implementation and robust-sigmoid utilities.

## Expected data layout

Place local data and generated outputs next to this repository root:

```text
data/
outputs/
```

Most notebooks were originally run in an internal project workspace. Absolute local paths have been replaced with generic placeholders such as `/path/to/sinkhole-risk-china`. Update those placeholders, or set up the same `data/` and `outputs/` structure before rerunning the notebooks.

## Python environment

The core workflow uses Python 3.9+ with geospatial and machine-learning packages. A minimal dependency list is provided in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some point-generation notebooks use `arcpy`, which requires an ArcGIS Pro Python environment and cannot be installed from PyPI.

## Typical workflow

1. Generate or prepare grid/sample points under `code/1_point_generation/`.
2. Extract the 12-factor predictor table with `code/2_feature_extraction/`.
3. Train and calibrate the national GWR classifier in `code/3_gwr_model_train/`.
4. Generate susceptibility predictions with `code/5_gwr_model_prediction/`.
5. Run attribution workflows under `code/4_attribution/`.
6. Estimate climate-change effects under `code/5_1_climate_change/`.
7. Run mitigation optimization under `code/6_NSGA_II_optimization/`.

## Notes

- Notebook outputs were stripped for a clean code release.
- Large generated artifacts are intentionally ignored by `.gitignore`.
- No software license has been selected yet; add one before making this repository public if reuse terms need to be explicit.
