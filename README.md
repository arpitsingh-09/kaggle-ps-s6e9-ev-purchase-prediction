# Kaggle PS-S6E9: Predicting Electric Vehicle Purchases

Logistic regression with cross-fitted target encoding for the
[Kaggle Playground Series, Season 6, Episode 9](https://www.kaggle.com/competitions/playground-series-s6e9).

The goal is to predict the probability that a person will buy an electric vehicle (`Will_Buy_EV`).
The metric is **ROC AUC**.

> **Credit:** Based on the public notebook *"KPS6E09 Logistic Regression Sample"*.
> Original author: `<add author name and link here>`.

---

## Overview

Instead of using a complex model such as LightGBM, this project uses a plain
**logistic regression** and puts the effort into **feature engineering**. A simple,
fast, interpretable model is trained on several hundred carefully built features.

## Dataset

| File | Description |
|------|-------------|
| `train.csv` | Training data with the target `Will_Buy_EV` (Yes/No) |
| `test.csv` | Test data without the target |
| `EV_Adoption_and_Range_Anxiety_Dataset.csv` | Original real-world dataset the competition data was generated from |

**Features**

- Numeric: `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Environmental_Concern_Level`,
  `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`
- Categorical: `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`,
  `Subsidy_Available`, `Range_Anxiety_Level`

> The data is **not included** in this repository. Please download it from the
> [competition page](https://www.kaggle.com/competitions/playground-series-s6e9/data)
> and from the original dataset's Kaggle page, and follow the competition rules.

## Approach

1. **Data validation:** checks for unique IDs, missing values, unknown categories,
   and overlap between train, test, and the original data.
2. **Features without the target:**
   - raw columns
   - digits of income and commute distance
   - a hand-made `BUY_SCORE` (income, concern level, subsidy, range anxiety)
   - range-anxiety features estimated from commute, charging stations, home charging, and city type
   - frequency counts and income priors from the original dataset
3. **Features with the target (cross-fitted):**
   - target encoding of income, commute, and number-prefix groups (as logits)
   - smoothed "neighbor" buy rates for similar incomes and commutes
   - interactions between concern level, subsidy, and anxiety
   - a small mixture model and an additive logistic model whose outputs become features
4. **Leakage control:** all target-based features are cross-fitted
   (10 outer folds, 5 inner folds), so a row's features never use its own label.
5. **Model:** `LogisticRegression(C=0.1, solver="lbfgs")` on standardized features,
   one model per fold.
6. **Prediction:** the test prediction is the average of the 10 fold models.

## Results

| Metric | Value |
|--------|-------|
| Pooled out-of-fold ROC AUC | `<fill in after running>` |
| Mean fold AUC | `<fill in after running>` |

## Outputs

- `submission.csv`: final predictions (`id`, `Will_Buy_EV`)
- `oof/LOGREG_SAMPLE_OOF.parquet`: out-of-fold predictions (useful for blending)
- `test_preds/LOGREG_SAMPLE_TEST.parquet`: test predictions
- `artifacts_for_analysis/`: run summary, timings, checks, coefficients, and feature importance

## How to run

1. Download the data from Kaggle and place the files in a `data/` folder.
2. In the notebook, set `IS_KAGGLE = False` so the paths point to `data/`.
3. Install the requirements:

   ```bash
   pip install numpy pandas scipy scikit-learn numba threadpoolctl pyarrow
   ```

4. Run the notebook `kps6e09-logistic-regression-sample.ipynb` from top to bottom.

Notes:

- The first cell enables GPU acceleration (`cudf.pandas`, `cuml.accel`) for Kaggle.
  Remove it if you run on a machine without an NVIDIA GPU.
- For a quick test, set the environment variable `LOGREG_SAMPLE_SMOKE=1` to run on a small sample.

## Repository structure

```
.
├── kps6e09-logistic-regression-sample.ipynb
├── README.md
├── data/                  # not included, download from Kaggle
├── oof/                   # generated
├── test_preds/            # generated
└── artifacts_for_analysis/  # generated
```

## Key learnings

- Cross-fitting is essential when using target encoding, or the model overfits.
- Synthetic data can contain patterns (such as number digits) that real data does not.
- A well-engineered linear model can be competitive with tree-based models on tabular data.

## License and credits

- Competition: Yao Yan, Walter Reade, Elizabeth Park. *Predicting Electric Vehicle Purchases.*
  Kaggle, 2026. https://www.kaggle.com/competitions/playground-series-s6e9
- Code: `<add your license, e.g. MIT, and respect the original notebook's license>`
