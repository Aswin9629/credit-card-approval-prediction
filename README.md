# Credit Card Approval Prediction

An academic classification study using application attributes and repayment history to explore credit-risk labeling, preprocessing and model comparison.

**Stack:** Python · pandas · scikit-learn · LightGBM · CatBoost · Matplotlib · Seaborn  
**Status:** Analysis prototype. The label is derived from repayment history; it is not an observed lender approval decision.

[Analysis source](CREDIT%20CARD%20APPROVAL%20PREDICTION.py) · [Dependencies](requirements.txt) · [Data instructions](data/README.md)

## Workflow

1. Read application and monthly credit-history records.
2. Map `X` and `C` status values to zero and flag applicants with a nonzero repayment status.
3. Join the datasets by applicant ID and remove duplicate records.
4. Fill missing occupation values with `Other` and explore demographic/financial attributes.
5. Scale numerical features and one-hot encode categorical features through a `ColumnTransformer`.
6. Compare classifiers using accuracy and classification reports.

The source includes Random Forest, Decision Tree, AdaBoost, KNN, Logistic Regression, LightGBM, CatBoost, Extra Trees, Gaussian Naive Bayes and Gradient Boosting experiments.

## Data requirements

Place both files in the repository root:

| File | Availability |
| --- | --- |
| `credit_record.csv` | Included |
| `application_record.csv` | Required separately; not included in this repository |

The application file must contain `ID` and the demographic/financial columns referenced in the script. The credit file uses `ID`, `MONTHS_BALANCE` and `STATUS`.

Install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

The original analysis is exported as a Python script:

```bash
python "CREDIT CARD APPROVAL PREDICTION.py"
```

## Reproduction notes

The export needs a few repairs before a reliable end-to-end rerun: import `FunctionTransformer` for the Naive Bayes section; handle sparse and dense matrices consistently; restrict the correlation heatmap to numeric columns with current pandas versions; and replace chained dataframe assignments with explicit operations.

The current join can leave several monthly records for an applicant, after which the script drops IDs and performs a random row split. A defensible evaluation should preserve applicant IDs and split by applicant (or aggregate first) to avoid the same person appearing in training and testing.

## Evaluation boundaries

The repository does not include a validated benchmark report, saved model or deployment. Accuracy alone is insufficient for an imbalanced target; future evaluation should report per-class precision/recall, PR-AUC, calibration and subgroup performance on an applicant-level holdout.

This is a learning project, not a lending decision system. Demographic attributes and retrospective target construction require careful review before any real-world use.

## Author

**Aswin S** · [GitHub](https://github.com/Aswin9629)
