# Notebooks

## Framework overview and approximate Gaussian process

The notebook `OverviewFigure_GP_Figure.ipynb` uses mechanistic models of infectious disease to simulate and constrain relative fitness.
The approximate Gaussian process model is demonstrated on data from these simulations.
This notebook generates figures:
- `../manuscript/figures/mechanism_identification.png`
- `../manuscript/supplementary_figures/gp_example.png`.

## Determining the transmisibility-escape tradeoff

The notebook `TransmissibilityEscapeTradeoff.ipynb` shows the trade-off between mechanisms of increased transmissibility and immune escape for relative fitness.
This generates `../manuscript/figures/transmission_tradeoff.png`.

##  Initial growth rates insufficient for predicting short-term frequency growth

The notebook `ShortTermDivergenceByMechanism.ipynb` shows how different transmission mechanisms can lead to large transmsission differences in the short-term and can limit the ability for short-term forecasts to reliably predict growth.
This generates `../manuscript/supplementary_figures/short_term_divergence.png`.

## Correlations insufficient for mechanism identification

The notebook `InterpretingImmuneEscapeFromRelativeFitness.ipynb` illustrates the limitations in using correlations between immunity (as quantified through vaccination uptake) and variant fitness it.
It generates `../manuscript/figures/mechanism_identification.png`.

## Quantifying selective pressure and predicted epidemic growth rates

The notebook `PreparingSelectivePressureDataSet.ipynb` estimates the relative fitness using the approximate Gaussian process model for US states and several countries on Nextstrain clade-level sequence counts, computes selective pressure, and computes epidemic growth rates for later model development in `SelectivePressureModelComparison.ipynb`.

This notebook generates figures:
- `../manuscript/supplementary_figures/selective-pressure-analysis_group_1.png`
- `../manuscript/supplementary_figures/selective-pressure-analysis_group_2.png`
- `../manuscript/supplementary_figures/selective-pressure-analysis_group_3.png`
- `../manuscript/supplementary_figures/selective-pressure-analysis_group_4.png`
- `../manuscript/supplementary_figures/selective-pressure-analysis_group_5.png`
- `../manuscript/supplementary_figures/selective-pressure-analysis_group_6.png`

The notebook `SelectivePressureModelComparison.ipynb` uses time-series cross-validation to compare several models for predicting epidemic growth rates from selective pressure.
It generates figures:
- `../manuscript/figures/selective_pressure_prediction.png`
- `../manuscript/supplementary_figures/empirical_growth_rate_predictions_all.png`
- `../manuscript/supplementary_figures/growth-rate-prediction-model-comparison.png`

## Latent immune factor model of relative fitness

The notebook `LatentFactorModel.ipynb` uses the latent factor model to estimate relative fitness for several countries at Pango lineage-level sequence counts, does comparison among different choices of latent dimension $D$, and analyzes the results of these models showing correlations between the titer data and estimated latent immune distances.
It generates figures:
- `../manuscript/figures/latent_immune.png`
- `../manuscript/supplementary_figures/loss_by_latent_dimension.png`
- `../manuscript/supplementary_figures/pseudo_escape_titer_correlation_bootstrap.png`
- `../manuscript/supplementary_figures/titer_pseudo_escape_distance_by_group.png`
- `../manuscript/supplementary_figures/latent_immune_2_dims.png`
- `../manuscript/supplementary_figures/latent_immune_4_dims.png`
- `../manuscript/supplementary_figures/latent_immune_6_dims.png`
- `../manuscript/supplementary_figures/latent_immune_8_dims.png`
