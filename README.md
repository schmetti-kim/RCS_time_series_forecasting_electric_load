# RCS_time_series_forecasting_electric_load

Current Progress:
<img width="850" height="206" alt="image" src="https://github.com/user-attachments/assets/12569a73-a844-4a29-aa49-c083b736b03c" />
(+ post-hoc residual analysis/correction complete)


## Methodology Overview

### 1. Univariate Forecasting (Baseline)
- Chronos-2 evaluated in a pure zero-shot setting using only historical load observations (no covariates, no related-region data).
- 7-day hourly context window (168 obs.) → 1-day forecast horizon (24 obs.), rolling one day forward across a 3-year evaluation period (1,095 forecasting cases per dataset).
- Establishes the performance floor against which all later experiments are compared.
- Result: clear performance ordering across datasets — **Latvia < Panama < South Australia** (lowest to highest error), consistent with the underlying load distributions (Latvia: low mean/variance, near-normal; Panama: multimodal; South Australia: high mean/variance, right-skewed).

### 2. Covariate-Informed Forecasting
- Extends the baseline by feeding Chronos-2 exogenous variables alongside load history, using its native covariate/group-attention support.
- 8 covariates selected from 17 candidates based on frequency of use in the load-forecasting literature: day of week, holiday, weekend (calendar); temperature, dew point, mean sea-level pressure, low cloud cover, wind speed at 10m (weather).
- Evaluated in 3 progressive stages:
  1. Calendar variables only
  2. Calendar + weather observations in the context window
  3. Calendar + weather in both context window **and** forecast horizon (using actual future weather as an upper-bound estimate, since forecast weather wasn't consistently available for the full 3-year period)
- **Result:** Best config across all 3 datasets = `holiday + weekend + temperature`. Significantly outperforms the univariate baseline everywhere (Panama p=0.0116, South Australia & Latvia p<0.001, via Diebold–Mariano test). South Australia benefits most; Latvia mostly from calendar info alone; Panama only from horizon weather.

### 3. Cross-Learning
- Tests whether feeding Chronos-2 load data from a *related* region (alongside the target region) improves forecasts, using its cross-series information-sharing capability.
- Related regions chosen for physical/market grid interconnection, not just proximity: Panama↔Costa Rica (SIEPAC interconnection), South Australia↔Victoria (Heywood interconnector, shared NEM), Latvia↔Lithuania (shared Baltic grid, now EU-synchronized).
- Uses the best covariate config (`holiday + weekend + temperature`) from stage 2 as the base.
- **Result:** Cross-learning *hurts* performance for Panama and South Australia (p<0.001 and p=0.004) relative to covariate-informed alone. Latvia sees a small, non-significant improvement (p=0.28).
- **Holiday-based filtering refinement:** excludes the related-region series from a case whenever the target and related region have different holiday statuses (affects only 0.41–2.24% of cases). This meaningfully shifts the loss differences and makes Latvia's cross-learning advantage statistically significant (p=0.0135); for Panama/SA it narrows but doesn't erase the disadvantage.

### 4. Residual Analysis & Correction
- Residuals from the best Chronos-2 configuration are examined for systematic structure: overall distribution (~normal, centered at 0), and MAPE conditioned on hour-of-day and holiday status (both show clear patterns — e.g. higher MAPE on holidays and during midday hours).
- Linear regression of residuals on available covariates explains very little variance (R² = 0.076–0.097 across datasets) → residual structure is likely nonlinear.
- **XGBoost residual correction:** trained on 70% of evaluation cases to predict residuals from covariates, applied to the held-out 30% to correct the original Chronos-2 forecasts.
- **Result:** Reduces mean RMSE and MAPE across all 3 datasets, with a more pronounced drop in variance; Diebold–Mariano confirms significantly lower loss (p well below 0.05) for all datasets. Caveat: only ~329 evaluation cases per dataset (vs. 1,095 in the main experiments), so results are preliminary and need validation on larger samples.

### Datasets
| Role | Dataset | Region | Level | Period |
|---|---|---|---|---|
| Target | Panama | Central America | Country | 2015–2018 |
| Target | South Australia | Australia | State | 2010–2013 |
| Target | Latvia | Europe | Country | 2021–2024 |
| Related | Costa Rica | Central America | Country | 2015–2018 |
| Related | Victoria | Australia | State | 2010–2013 |
| Related | Lithuania | Europe | Country | 2021–2024 |

## Error Progression Across Experiments

**Mean RMSE**
![Relative Mean RMSE Change vs. Univariate](data/Images/mean_rmse.png)

**Variance RMSE**
![Relative Variance RMSE Change vs. Univariate](data/Images/variance_rmse.png)

**Mean MAPE**
![Relative Mean MAPE Change vs. Univariate](data/Images/mean_mape.png)

**Variance MAPE**
![Relative Variance MAPE Change vs. Univariate](data/Images/variance_mape.png)

| Dataset | Mean RMSE | Variance RMSE | Mean MAPE | Variance MAPE |
|---|---|---|---|---|
| **Panama** | 51.23 → 49.88 → 51.36 → **51.28** (net ~0%) | 1757 → 1610 → 1711 → **1700** (net −3.3%) | 3.55 → 3.47 → 3.57 → **3.56** (net ~0%) | 8.55 → 7.88 → 8.34 → **8.25** (net −3.5%) |
| **South Australia** | 103.54 → 82.89 → 85.08 → **84.85** (net **−18.1%**) | 9825 → 4282 → 4901 → **4833** (net **−50.8%**) | 6.00 → 4.90 → 5.01 → **4.99** (net **−16.9%**) | 29.00 → 13.28 → 15.11 → **14.60** (net **−49.7%**) |
| **Latvia** | 27.59 → 24.84 → 24.69 → **24.57** (net **−11.0%**) | 432.69 → 349.41 → 340.38 → **327.78** (net **−24.2%**) | 2.92 → 2.65 → 2.63 → **2.62** (net **−10.2%**) | 5.88 → 4.68 → 4.65 → **4.47** (net **−23.9%**) |

**Key findings:**

- **Covariates deliver the single biggest improvement across the board.** The Univariate → Covariate step accounts for nearly all of the total error reduction — Australia's variance drops ~54% and Latvia's ~20% right at this step.
- **Cross-learning erases some of that gain for Panama and Australia.** Both see mean RMSE/MAPE tick back up slightly once cross-learning is introduced — consistent with the DM test showing cross-learning is significantly worse than covariate-informed alone for these two.
- **Latvia is the outlier — errors keep falling monotonically through every stage**, including cross-learning, and it's the one dataset where holiday-based filtering shows its clearest payoff.
- **Holiday-based filtering nudges things in the right direction for everyone**, even where it can't fully recover the covariate-informed baseline (Panama, Australia).
- **Variance shrinks far more than the mean does, almost everywhere** — e.g. Australia's variance drops ~50–56% vs. a ~17–20% mean drop — meaning these strategies improve forecast *consistency* more than average accuracy.
- **Panama is the most stable dataset** — its mean error barely moves after the covariate stage, suggesting the improvement ceiling for these strategies was mostly reached early.
