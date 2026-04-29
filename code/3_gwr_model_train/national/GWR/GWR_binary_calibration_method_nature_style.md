# Post-hoc Calibration of GWR Scores for Binary Classification

## 1. What can be claimed rigorously

Yes, the current binary workflow has a defensible statistical logic, but it should be described precisely.

The rigorous statement is:

`The raw GWR regression output was treated as a continuous risk score and then converted to an estimated event probability using a post-hoc logistic calibration model after robust scaling.`

The statement that should be avoided is:

`The raw GWR regression output itself is a probability.`

This distinction matters because the empirical GWR output is not naturally restricted to the interval `[0,1]`. In the present dataset, the score distribution is heavy-tailed and extends far below `0` and above `1`. Direct truncation,

$$
p_i^{\mathrm{clip}} = \min\{1, \max(0, s_i)\},
$$

is not a likelihood-based probability model, does not use the observed labels, and collapses all values below `0` to `0` and all values above `1` to `1`. That operation destroys information in the tails and can severely distort both discrimination and calibration.

By contrast, the calibration step used here is a standard post-hoc probabilistic mapping: the GWR output is first treated as a one-dimensional score, and a monotonic logistic model is then fitted to estimate

$$
P(Y_i = 1 \mid s_i).
$$

This is statistically coherent and is closely related to one-dimensional logistic recalibration or Platt-style scaling.

## 2. Mathematical formulation

Let:

- $s_i$ be the raw GWR prediction score (`predy`) for sample $i$.
- $y_i$ be the observed response.
- $y_i^{(b)}$ be the derived binary label.

### 2.1 Binary target definition

The binary response is defined as

$$
y_i^{(b)} = \mathbb{I}(y_i > c),
$$

where $\mathbb{I}(\cdot)$ is the indicator function and $c = 0.55$ in the current implementation.

### 2.2 Robust scaling of the GWR score

Because the raw GWR scores are heavy-tailed, they are standardized using the median and interquartile range estimated on the fitting subset:

$$
z_i = \frac{s_i - \mathrm{median}(S_{\mathrm{fit}})}{\mathrm{IQR}(S_{\mathrm{fit}})}.
$$

This step reduces the influence of extreme values and makes the subsequent logistic fit numerically stable.

### 2.3 Logistic calibration

The calibrated event probability is modeled as

$$
p_i = P(Y_i = 1 \mid s_i) = \sigma(\beta_0 + \beta_1 z_i),
$$

where

$$
\sigma(x) = \frac{1}{1 + e^{-x}}.
$$

Substituting the robust scaling step gives

$$
p_i = \frac{1}{1 + \exp\left[-\left(\beta_0 + \beta_1
\frac{s_i - \mathrm{median}(S_{\mathrm{fit}})}{\mathrm{IQR}(S_{\mathrm{fit}})}\right)\right]}.
$$

This is a monotonic transformation of the original GWR score, so the ordering induced by $s_i$ is preserved when $\beta_1 > 0$.

### 2.4 Threshold selection for binary classification

The final binary decision is not made on the raw GWR score, but on the calibrated probability:

$$
\hat{y}_i = \mathbb{I}(p_i \ge t^*).
$$

The decision threshold $t^*$ is selected on a held-out validation subset by maximizing Youden's index:

$$
t^* = \arg\max_t J(t), \qquad J(t) = \mathrm{TPR}(t) - \mathrm{FPR}(t).
$$

This gives a threshold that balances sensitivity and specificity under the ROC framework.

## 3. Fitted form for the current model

For the current national-scale implementation, the fitted parameters are:

- `median(S_fit) = 0.54319589`
- `IQR(S_fit) = 0.93264739`
- `beta_0 = 0.18762438`
- `beta_1 = 8.10789288`
- `t* = 0.68098801`

Therefore, the calibrated probability can be written explicitly as

$$
p_i =
\frac{1}{1 + \exp\left[-\left(0.18762438 +
8.10789288 \cdot \frac{s_i - 0.54319589}{0.93264739}\right)\right]}.
$$

An equivalent form directly in terms of the raw GWR score is

$$
p_i =
\frac{1}{1 + \exp\left[-\left(-4.53460412 + 8.69341722\, s_i\right)\right]}.
$$

The final binary decision rule is

$$
\hat{y}_i = \mathbb{I}(p_i \ge 0.68098801).
$$

Because the fitted mapping is monotonic increasing, this is equivalent to a raw-score threshold:

$$
\hat{y}_i = \mathbb{I}(s_i \ge 0.60884230).
$$

The current validation performance is:

- `Accuracy = 0.9338`
- `ROC-AUC = 0.9835`
- `Youden's J = 0.8677`
- `TPR = 0.9192`
- `FPR = 0.0516`

## 4. Nature-style Methods text in English

### Binary classification from GWR scores

The raw GWR prediction output was not interpreted as a probability because its empirical distribution extended well beyond the unit interval. We therefore treated the GWR output as a continuous risk score and applied a post-hoc calibration step to estimate event probabilities. Let $s_i$ denote the GWR prediction score for location $i$. The reference binary outcome was defined from the observed response using a fixed cutoff $c = 0.55$, such that $y_i^{(b)} = \mathbb{I}(y_i > c)$. To reduce the influence of the heavy-tailed score distribution, the GWR score was robustly standardized using the median and interquartile range estimated on the fitting subset, $z_i = \{s_i - \mathrm{median}(S_{\mathrm{fit}})\}/\mathrm{IQR}(S_{\mathrm{fit}})$. A one-dimensional logistic calibration model was then fitted as $P(Y_i = 1 \mid s_i) = \sigma(\beta_0 + \beta_1 z_i)$, where $\sigma(x) = (1 + e^{-x})^{-1}$. This formulation provides a statistically principled mapping from the GWR score to the probability scale while preserving the score ranking through a monotonic link function.

### Model fitting and threshold selection

Calibration was fitted on `80%` of the available samples and evaluated on a stratified `20%` validation subset (`random_state = 42`). The final binary decision threshold was selected on the validation subset by maximizing Youden's index, $J(t) = \mathrm{TPR}(t) - \mathrm{FPR}(t)$, over the calibrated probabilities. Samples were assigned to the positive class when $p_i \ge t^*$. In the present implementation, the fitted parameters were $\mathrm{median}(S_{\mathrm{fit}}) = 0.5432$, $\mathrm{IQR}(S_{\mathrm{fit}}) = 0.9326$, $\beta_0 = 0.1876$ and $\beta_1 = 8.1079$, giving

$$
p_i =
\frac{1}{1 + \exp\left[-\left(0.1876 +
8.1079 \cdot \frac{s_i - 0.5432}{0.9326}\right)\right]}.
$$

The optimal validation threshold was $t^* = 0.6810$. Under this protocol, the validation accuracy was `0.9338` and the validation ROC-AUC was `0.9835`.

## 5. Recommended wording in Chinese for internal documentation

This study does not interpret the raw regression output of GWR directly as a probability, but first treats it as a continuous risk score, which is then mapped into an event probability through a robustly scaled logistic calibration model. This process essentially belongs to posterior probability calibration, rather than rewriting the GWR regression model itself into a probability model. Its theoretical basis is that if there is a monotonic relationship between the original score and event risk, univariate logistic mapping can transform the score into the `[0,1]` probability space while retaining the ranking information; while using the median and interquartile range for robust scaling, it can reduce the impact of extreme values ​​on parameter estimates. The final binary classification threshold is determined by Youden's index on the validation set, rather than directly setting an empirical threshold on the raw GWR score or the clipped pseudo-probability.

## 6. Important caution for manuscript writing

If the manuscript explicitly claims that the probabilities are `well calibrated`, this should ideally be supported by additional calibration diagnostics, such as a reliability curve, Brier score, calibration intercept and slope, or expected calibration error. The current workflow is already theoretically valid as a post-hoc calibration model, but a stronger calibration claim should be backed by explicit calibration evidence.
