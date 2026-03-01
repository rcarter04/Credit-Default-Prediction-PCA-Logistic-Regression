# Credit-Default-Prediction-PCA-Logistic-Regression
Predicting Customer Credit Default: A Comparative Statistical Study on PCA and Logistic Regression

### Project Overview
This repository contains the full analytical workflow for classifying credit risk using the German Credit Dataset. 
The study employs **Principal Component Analysis (PCA)** for dimensionality reduction and **Logistic Regression** for classification.

### File Descriptions
* **german_credit_data.csv**: The raw dataset containing 1,000 instances and 20 features.
* **Credit_Default_Risk_Modeling.ipynb**: Jupyter Notebook containing data cleaning, log-transformation, PCA, and model evaluation.
* **Term_Paper.pdf**: The comprehensive APA 7th edition report.

## Key Statistical Findings

### 1. Data Distribution & Transformation
- **Class Imbalance:** The dataset consists of 70% Good Credit (0) and 30% Bad Credit (1) risks.
- **Normality:** Shapiro-Wilk tests and Q-Q plots confirmed that numerical features like `credit_amount` and `age` were non-Gaussian. 
- **Preprocessing:** Applied `log1p` transformation and `StandardScaler` to stabilize variance and normalize feature scales.

### 2. Dimensionality Reduction (PCA)
- **Feature Expansion:** One-hot encoding expanded the initial categorical features into 61 binary columns.
- **Variance Retention:** PCA was configured to retain **95% of total variance**, successfully reducing the dimensionality from **61 features to 32 principal components**.

### 3. Model Performance (Logistic Regression)
The model was evaluated on a 20% hold-out test set with the following results:
- **Test Accuracy:** 77.00%
- **AUC-ROC:** 0.80 (Demonstrating strong discriminatory power)
- **Precision (Class 1):** 0.63
- **Recall (Class 1):** 0.53

### 4. Business Implications (Threshold Analysis)
A key part of this study involved a **Threshold Experiment**. By lowering the classification threshold, we prioritized **Recall over Precision**. 
- **Insight:** In a credit risk scenario (similar to a medical diagnosis), missing a "Bad" risk (False Negative) is significantly more expensive than misclassifying a "Good" risk (False Positive).
- **Outcome:** The study demonstrates how statistical models can be tuned to align with a financial institution's specific risk tolerance.

### Dependencies
* Python 3.x
* Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
