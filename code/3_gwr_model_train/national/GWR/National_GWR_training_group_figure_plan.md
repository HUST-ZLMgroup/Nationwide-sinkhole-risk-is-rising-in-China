# National GWR Training Group Figure Plan

## Target

Design one Nature Communications-style group figure to summarize the national GWR threshold diagnostics, training-sample probability structure and predictive accuracy for sinkhole collapse probability.

Recommended figure title:

**Figure X. Threshold diagnostics, training probability structure and validation of the national GWR model for sinkhole probability.**

Recommended Chinese title:

**Figure X. Threshold diagnosis, training probability structure and verification of the national ground collapse probability GWR model. **

## Shared Style Rules

- Output format: `SVG`
- Background: pure white
- Font: Times New Roman
- Base font size: 9 pt
- Panel labels: bold, 14 pt
- Color system: low-saturation Morandi palette
- Avoid heavy borders, dense legends and large explanatory text blocks
- Keep legends as glyph-based annotations where possible
- Preserve editable text in SVG

Suggested palette:

| Role | Color |
|---|---|
| Hydrogeology | `#7E8FA3` |
| Climate | `#9AAF9A` |
| Anthropogenic | `#C69074` |
| Calibration / probability | `#8D6E63` |
| Validation / uncertainty | `#7F8F9E` |
| Positive class / high risk | `#C69074` |
| Negative class / low risk | `#8CA0B3` |
| Grid / guide lines | `#E6E2DD` |
| Text secondary | `#6F6A64` |

## Recommended Layout

Use a 180 mm wide group figure with 5 panels.

Recommended canvas:

- Width: 180 mm
- Height: approximately 120 mm
- Grid: independent top and bottom subgrids
- Top row: panel `a` and panel `b` use an approximate `43:57` width ratio, giving more space to threshold diagnostics.
- Bottom row: panels `c`, `d` and `e` each span two columns.
- Horizontal spacing is intentionally compact to reduce white space between panels.

Panel organization:

| Panel | Role | Main message |
|---|---|---|
| a | Threshold diagnostics | Sensitivity, specificity and F1 are evaluated across probability cutoffs, with the standard probability threshold marked at 0.500. |
| b | Training probability distribution | The calibrated probabilities of the 25,232 labelled historical training samples are partitioned into five susceptibility classes using Jenks natural breaks. |
| c | Repeated validation performance | Cross-validation shows stable discrimination and classification performance. |
| d | ROC uncertainty | The model has strong ranking ability across repeated validation folds. |
| e | Probability reliability | Calibrated probabilities remain close to empirical event frequencies, with low Brier error. |

## Panel Design

### a. Threshold Selection Curve

Plot form:

- Multi-curve threshold diagnostic with decision threshold on the x-axis.
- Plot sensitivity, specificity and F1.
- Mark the standard calibrated probability cutoff `p = 0.500`.

Data source:

- `gwr_classification_artifacts.pkl`
- `gwr_model_national_GWR.pkl` or training `gwr_results.predy`
- `gwr_classification_artifacts.pkl`

Key values to annotate:

- Transform: `robust_sigmoid_iqr`
- Probability threshold shown in the figure: `0.500`
- Curves are calculated on the validation subset.

Main message:

The threshold diagnostic shows how validation sensitivity, specificity and F1 vary after raw GWR scores have been mapped to probabilities. The plotted operating cutoff is the standard probability threshold of 0.500.

### b. Training Probability Distribution

Plot form:

- Probability-density diagram of the labelled historical training samples.
- Partition the calibrated training probabilities into five susceptibility classes using Jenks natural breaks.
- Annotate class percentages and the total number of training samples.
- Keep the five class labels horizontal above the density bands.

Data source:

- `data/Extracted_HAVE_future/Positive_Negative_balanced/AllFeatures_Positive_Negative_balanced_25366_ssp1_cleaned.csv`
- `gwr_classification_artifacts.pkl`
- `gwr_model_national_GWR.pkl`

Key values:

- Full labelled historical training dataset: `n = 25,232`
- Input table: positive-negative balanced training CSV
- Probability range: approximately `0.072-0.850`
- Jenks natural breaks: approximately `0.072`, `0.282`, `0.415`, `0.556`, `0.681`, `0.850`
- Class shares: approximately `35.9%`, `9.4%`, `8.0%`, `11.9%`, `34.8%`

Main message:

The probability-class structure is derived from the labelled historical training dataset, not from the national prediction grid. This panel should use `AllFeatures_Positive_Negative_balanced_25366_ssp1_cleaned.csv`, not `AllFeatures_Points_China_all_*_cleaned.csv`.

### c. Repeated Validation Metric Constellation

Plot form:

- Dot-interval or half-violin metric constellation.
- Show train and test distributions side-by-side for each metric.
- Use test metrics as the visual focus; train metrics should be lighter.

Data source:

- `national_fig_gwr/gwr_classification_metrics_rskf5x10_boot1000.csv`
- Preferred protocol: repeated stratified 5-fold cross-validation, 10 repeats, 50 validation folds.

Recommended metrics:

- Accuracy
- Balanced accuracy
- F1
- ROC-AUC
- Average precision
- Brier score

Approximate test performance from repeated validation:

- Accuracy: `0.908 +/- 0.004`
- Balanced accuracy: `0.908 +/- 0.004`
- F1: `0.910 +/- 0.003`
- ROC-AUC: `0.968 +/- 0.002`
- Average precision: `0.962 +/- 0.003`
- Brier score: `0.068 +/- 0.002`

Main message:

Repeated validation indicates stable out-of-sample performance, with high discrimination and low probability error.

### d. ROC Uncertainty

Plot form:

- ROC curve fan.
- Thin transparent curves for individual folds.
- Darker mean curve with a small text annotation.
- No large legend; use direct labels.

Data source:

- `national_fig_gwr/_cache/gwr_cls_rskf5x10_boot1000_seed2026_debug1.pkl.gz`

Main message:

The model maintains strong ranking performance across repeated validation folds, not only in a single split.

### e. Probability Reliability

Plot form:

- Reliability curve with 1:1 reference line
- Point size can represent the average number of samples in each probability bin
- Annotate the selected probability threshold and mean Brier score

Data source:

- `national_fig_gwr/_cache/gwr_cls_rskf5x10_boot1000_seed2026_debug1.pkl.gz`
- `national_fig_gwr/gwr_classification_metrics_rskf5x10_boot1000.csv`

Key values:

- Probability threshold: `0.500`
- Mean test Brier score: approximately `0.068`

Main message:

The calibrated probability provides useful probabilistic information, not only class ranking.

## Recommended Caption

**Figure X. Threshold diagnostics, training probability structure and validation of the national GWR model for sinkhole probability.** a, Validation threshold diagnostics after robust sigmoid calibration of raw GWR scores. Sensitivity, specificity and F1 are shown across candidate decision thresholds, and the standard probability cutoff is marked at 0.500. b, Calibrated probability distribution of the 25,232 labelled historical training samples from the positive-negative balanced dataset. The distribution is partitioned into five susceptibility classes using Jenks natural breaks fitted to the training-sample probabilities. c, Repeated validation metrics from stratified 5-fold cross-validation with 10 repeats. Points and intervals summarize the stability of out-of-sample performance. d, ROC uncertainty curves across repeated validation folds. e, Probability reliability curve showing the agreement between predicted probabilities and empirical event frequencies; the Brier score summarizes probabilistic error.

## Recommended Result Text

The national GWR model was trained on 25,232 balanced historical samples using 12 hydrogeological, climatic and anthropogenic predictors. Because the raw GWR output is a continuous regression score rather than a bounded probability, the score was transformed using a robust sigmoid calibration fitted on the training distribution. The calibrated training probabilities were then partitioned into five susceptibility classes using Jenks natural breaks, revealing a bimodal probability structure in the labelled training set. Validation threshold diagnostics show how sensitivity, specificity and F1 vary across cutoffs, with the standard calibrated probability threshold marked at 0.500.

Repeated validation showed stable predictive skill. Across stratified 5-fold cross-validation repeated 10 times, the test ROC-AUC was approximately 0.968, average precision was 0.962, accuracy was 0.908 and F1 was 0.910, with small fold-to-fold variation. The Brier score remained low at approximately 0.068, indicating that the calibrated probabilities retained useful probabilistic information in addition to high ranking performance. These results support the use of the trained GWR-calibration framework for national-scale sinkhole probability prediction and subsequent susceptibility classification.

## Implementation Notes

Recommended script location:

- `code/3_gwr_model_train/national/GWR/national_fig_gwr/plot_gwr_training_group.py`

Recommended output:

- `code/3_gwr_model_train/national/GWR/national_fig_gwr/final/gwr_training.svg`

Required inputs:

- `gwr_classification_artifacts.pkl`
- `national_fig_gwr/gwr_classification_metrics_rskf5x10_boot1000.csv`
- `national_fig_gwr/_cache/gwr_cls_rskf5x10_boot1000_seed2026_debug1.pkl.gz`
- `gwr_model_national_GWR.pkl`
