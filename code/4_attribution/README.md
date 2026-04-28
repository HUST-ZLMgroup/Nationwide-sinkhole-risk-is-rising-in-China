# Attribution Module Status

This folder contains several attribution-related branches, but the current
mainline interpretation for the paper should be treated as **LiNGAM-based**,
not `RF + LiNGAM`.

## Current mainline

- **Primary attribution method**: `LiNGAM`
- **Main working files**:
  - `national_fig_lingam/lingam_division.ipynb`
  - `national_fig_lingam/LiNGAM_sankey_updated_12vars_equalheight_v2.ipynb`
- **Preprocessing dependency**:
  - `national_shp_division/province_no_TW_AM_HK_geographical_division_shp.py`
  - `national_shp_division/csv_division_added.py`

## Important clarification

- The SHAP branch under `shap_division/` is a **RandomForest + TreeSHAP**
  workflow.
- The `GWR_SHAP_division_12vars_v2_outputs/` package is also **not a true GWR
  attribution model**. Its own method note already states that the numerical
  core remains `RF + TreeSHAP`.
- Therefore, the current attribution baseline should **not** be described as
  `RF + LiNGAM`. These are separate branches.

## Current project decision

For the ongoing "second conclusion attribution analysis":

- Use **LiNGAM** as the core attribution framework.
- Treat the RF/SHAP branch as **legacy or supplementary only**.
- Do not mix RF importance and LiNGAM causal effects into a single "base model"
  description.

## Practical implication

When we continue editing this module, the default priority should be:

1. LiNGAM data pipeline
2. LiNGAM regional / national figures
3. LiNGAM Sankey outputs

RF/SHAP files are retained for reference, but they are **not** the primary
method path anymore.
