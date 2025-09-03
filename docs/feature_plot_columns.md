# Feature plot columns in the SOM notebook

Short summary of the change enabling feature plots for columns beyond the SOM training inputs.

## What changed
- Added `feature_plot_columns` next to `input_columns` in the Parameters cell. It defaults to a copy of `input_columns` and can include extra columns.
- Feature plots now iterate over `feature_plot_columns` instead of only `input_columns`.
- For columns not used to train the SOM, the notebook computes a per-cell mean from `df` (over samples mapped to each SOM cell) and renders that grid on the hex layout.
- Introduced a small helper `add_som_grid(...)` used by plotting to render arbitrary grids on the SOM.
- Existing behavior retained:
  - Specialized color scales are applied when a column name matches `specialized_colorscales`.
  - `force_feature_colorscale_to_01` controls whether color bars are fixed to [0, 1].
  - The Umatrix and the "Clusters and winning cells" panels are unchanged.

## How to use
1. In the Parameters cell, set the list of columns to visualize, e.g.:
   
   ```python
   feature_plot_columns = input_columns + ["CO3", "CH4"]
   ```
2. Re-run the notebook from data loading through the Feature Plots section.

## Notes
- SOM-weight-based plots use the learned weights for features in `input_columns`.
- Extra feature plots compute per-cell means with `pd.to_numeric(..., errors="coerce")`; cells with no samples are shown transparent.
- You can extend `specialized_colorscales` with entries for any new column names.

Files touched: `experimentation.ipynb`.
