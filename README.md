# Credit Card Approval Prediction

## Overview
This project focuses on predicting credit card approval risk using customer application records and credit history data. The goal is to classify applicants as good or risky clients based on their personal, financial, and credit behavior information.

The project combines two datasets: `application_record.csv` and `credit_record.csv`. A target variable was created from the credit status column, where customers with no overdue history were labeled as good clients and customers with at least one late payment record were labeled as risky clients.

## Key Features
- Loaded and merged application and credit record datasets
- Created a binary target variable from credit repayment status
- Removed duplicate records
- Handled missing occupation values by replacing them with `Other`
- Performed exploratory data analysis using visualizations
- Analyzed gender, car ownership, work phone, occupation, family status, housing type, family members, and education level
- Built preprocessing pipelines for numerical and categorical features
- Applied scaling and one-hot encoding
- Trained and compared multiple machine learning classification models
- Evaluated models using accuracy score and classification report

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- LightGBM
- CatBoost

## Dataset
The project uses two datasets:

- `application_record.csv`
- `credit_record.csv`

The application dataset contains customer demographic and financial details, while the credit dataset contains repayment status records. These datasets are merged using the customer ID.

## Machine Learning Models Used
- Random Forest Classifier
- Decision Tree Classifier
- AdaBoost Classifier
- K-Nearest Neighbors
- Logistic Regression
- LightGBM Classifier
- CatBoost Classifier
- Extra Trees Classifier
- Naive Bayes
- Gradient Boosting Classifier

## Features Used
### Numerical Features
- Number of children
- Total income
- Age / days birth
- Employment duration
- Number of family members

### Categorical Features
- Gender
- Car ownership
- Realty ownership
- Income type
- Education type
- Family status
- Housing type
- Occupation type

## Methodology
First, the credit status values were cleaned by replacing `X` and `C` with 0, representing good clients. Any customer with at least one late payment month was marked as 1, representing a risky client. The application and credit datasets were then merged using customer ID.

After cleaning and handling missing values, exploratory data analysis was performed using count plots, heatmaps, and distribution charts. A machine learning pipeline was created using `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder` to preprocess numerical and categorical features before model training.

Multiple classification models were trained and evaluated to compare their performance in predicting credit card approval risk.

## Visualizations
- Correlation heatmap
- Car ownership by gender
- Work phone availability by gender
- Occupation distribution
- Family status distribution
- Housing type distribution
- Family member count distribution
- Education level distribution
- Credit status by gender
- Gender distribution

## Future Improvements
- Add ROC-AUC, precision, recall, and F1-score comparison table
- Handle class imbalance using SMOTE or class weights
- Perform hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Add feature importance analysis
- Save the best model using Pickle or Joblib
- Build a Streamlit app for real-time credit approval prediction

## Author
Aswin S
