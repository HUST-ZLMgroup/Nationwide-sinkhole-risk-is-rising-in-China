# Source Data for Nature Communications submission

This folder contains the source-data package for the main-text figures.

The files are organized by figure (`Fig1` to `Fig8`). CSV files provide the numerical values underlying charts, rankings, attribution summaries, climate-counterfactual summaries and optimization results. GeoTIFF and shapefile components provide processed spatial layers for map-based panels. Fig. 2 model diagnostics are provided as open CSV/JSON exports, with the original small serialized caches retained for reproducibility.

Large third-party raw predictor datasets are not redistributed here. They are cited in the manuscript references and should be obtained from the original providers. Large gridded processed tables can be deposited separately in Figshare, Zenodo or another public data repository if the editor requests full processed data beyond the figure source data.

Suggested upload name: `Source_Data.zip`.

## Figure mapping

- `Fig1`: historical national susceptibility layer, provincial spatial summary and province-level spatial-count shapefile.
- `Fig2`: balanced training table, threshold-sensitivity data, calibrated probability and susceptibility-class summaries, repeated cross-validation metrics, ROC/reliability curve data, prediction records, GWR calibration metadata and susceptibility class boundaries.
- `Fig3`: national and provincial future relative-change summaries across SSP1-SSP5 and 2040-2100.
- `Fig4`: historical and SSP1-SSP5 2100 processed susceptibility GeoTIFFs.
- `Fig5`: historical LiNGAM total effects, adjacency matrix, edge list, pathway summary and variable dictionary.
- `Fig6`: future LiNGAM edge-frequency, stable-path and group-path summaries.
- `Fig7`: SSP2 climate-counterfactual metadata, processed climate-change GeoTIFFs and provincial summaries.
- `Fig8`: intervention strategy summaries, selected solution records and province-level cost/risk-reduction summaries.
