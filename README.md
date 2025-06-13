## Machine Learning Project: Financial Risk Assessment & Credit Card Fraud Detection
 ## Project Overview

This project involves applying machine learning techniques to two distinct datasets:

    Financial Risk Assessment – A regression problem aimed at predicting financial risk scores based on customer features.

    Credit Card Fraud Detection – A binary classification task focused on identifying fraudulent transactions from normal ones.

The objective is to explore, preprocess, and build appropriate models for each dataset, evaluate their performance, and understand their limitations.
Datasets
1. Financial Risk Assessment

    Samples: 2,000 rows, 37 columns.

    Target: RiskScore (Continuous variable).

    Task: Regression

2. Credit Card Fraud Detection

    Features: All numerical, derived from PCA transformation of original transaction data.

    Target: Class (0 = Legit, 1 = Fraud)

    Task: Binary Classification (heavily imbalanced)

## Preprocessing
Financial Risk Assessment:

    Missing Data: None

    Duplicate Entries: None

    Outliers: 94 detected in RiskScore using IQR.

    Scaling: MinMaxScaler for quantitative features.

    Encoding: LabelEncoder for categorical features like EmploymentStatus, EducationLevel.

    Feature Selection: Dropped ApplicationDate.

Fraud Detection:

    Class Imbalance: Resolved using Random Under-Sampling.

    Scaling: StandardScaler applied to all features.

    Data Cleaning: Removed duplicates and missing values.

## Models Implemented
 Financial Risk Assessment (Regression):
Model	Rationale

Linear Regression	Simple baseline, interpretable, performs well on linear data

Ridge Regression	Handles multicollinearity and reduces overfitting

KNN Regressor	Non-parametric, performance depends on k and scaling
## Evaluation Metrics:

    MSE (Mean Squared Error)

    RMSE (Root Mean Squared Error)

    MAE (Mean Absolute Error)

    R² Score
## Models Implemented
Credit Card Fraud Detection (Classification):
Model	Rationale

Logistic Regression	Interpretable and effective for linear problems

KNN Classifier	Captures local structure, but sensitive to scaling and imbalance
## Evaluation Metrics:

    Accuracy

    Precision

    Recall

    F1-Score

    Confusion Matrix

##  Results Summary
 Financial Risk Assessment:
Model	R² (Test)	MSE (Test)	MAE (Test)	RMSE (Test)
Linear Regression	0.9346	4.05	1.53	2.01
Ridge Regression	0.9347	4.06	1.53	2.01
KNN Regression	-0.0063	62.48	6.24	7.90

     Conclusion: Ridge regression slightly outperformed linear regression. KNN failed to model the data effectively, possibly due to complexity or lack of feature locality.

 Fraud Detection:
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	94.4%	1.00	0.90	0.95
KNN Classifier	89.8%	0.99	0.82	0.90

    Conclusion: Logistic Regression performed better with balanced metrics. KNN showed decent results but suffered from lower recall and higher sensitivity to class imbalance.

## Technologies Used

    Python

    Pandas, NumPy – Data manipulation

    Scikit-learn – Model building and evaluation

    Matplotlib, Seaborn – Visualizations

    Jupyter Notebook – Prototyping and experimentation

Limitations

    Financial Risk Assessment: Assumes linear relationships. Non-linear models (e.g., Random Forest, XGBoost) may yield better results.

    Fraud Detection: Under-sampling may lead to information loss from majority class, affecting generalization.
