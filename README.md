# Customer Churn Prediction

github Repo Link: https://github.com/NRAJKU/NAGP-assignment2-customer_data_churn

## 1. Project Overview

This project builds a machine learning solution to predict whether a telecom customer is likely to churn.

The business objective is to identify customers who are at higher risk of leaving so that the telecom company can take proactive retention actions such as targeted offers, service improvements, or customer outreach.

The solution covers the complete workflow:

**Business Problem → Data Understanding → Data Preparation → EDA → Feature Engineering → Model Development → Evaluation → Interpretation → Model Saving → REST API**

---

## 2. Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer demographic information, account information, subscribed services, billing information, and the target variable `Churn`.

- Target variable: `Churn`
- Target values: `Yes` / `No`
- Records: 7,043
- Columns: 21
- Duplicate rows: 0
- Churned customers: 1,869
- Non-churned customers: 5,174
- Churn rate: approximately 26.54%

`customerID` is treated as an identifier and is not used as a predictive feature.

---

## 3. Data Preparation

The notebook performs the following preparation steps:

1. Inspects the dataset structure and data types.
2. Checks for missing values and duplicate records.
3. Converts `TotalCharges` from string/object to numeric.
4. Separates numerical and categorical variables.
5. Excludes `customerID` from model features.
6. Creates reusable feature-engineering and preprocessing pipelines.
7. Applies median imputation to numerical features.
8. Applies most-frequent imputation to categorical features.
9. Applies one-hot encoding with `handle_unknown="ignore"`.
10. Splits the data into training and testing sets using:
   - 70% training data
   - 30% testing data
   - `random_state=42`
   - stratification by the target variable

Preprocessing is fitted as part of the scikit-learn pipeline on the training data, avoiding leakage from the test set.

---

## 4. Exploratory Data Analysis

The notebook includes visualizations covering:

- overall churn distribution;
- customer and service characteristics;
- churn by contract type;
- tenure distributions for churned and non-churned customers;
- monthly charges and other numerical relationships; and
- tenure versus monthly charges.

The EDA is used to identify patterns relevant to churn.

For example, contract type shows a strong association with churn, while shorter tenure and some service and billing characteristics are associated with higher churn.

These findings are interpreted as associations rather than causal relationships.

---

## 5. Feature Engineering

The project creates additional customer-level features that do not use the target variable.

### TotalServices

Counts the number of subscribed services across the available service-related fields.

This provides a simple measure of customer service engagement.

### IsNewCustomer

Identifies customers with tenure of 12 months or less.

This captures early-stage customers whose churn behavior may differ from longer-tenured customers.

### AvgMonthlySpend

Calculates average historical monthly spend as:

```text
TotalCharges / tenure
```

For customers with zero tenure, `MonthlyCharges` is used as the fallback.

These features are created independently of the target variable and are included in the model pipeline.

---

## 6. Model Development

The required model for the assignment is a Decision Tree Classifier.

Two initial configurations were evaluated.

### Model A

- Criterion: `gini`
- Maximum depth: `5`
- Minimum samples per leaf: `20`
- Class weight: `balanced`
- Random state: `42`

### Model B

- Criterion: `entropy`
- Maximum depth: `8`
- Minimum samples per leaf: `10`
- Class weight: `balanced`
- Random state: `42`

Both configurations use the same preprocessing and feature-engineering pipeline and are evaluated on the same held-out test set.

---

## 7. Hyperparameter Tuning

GridSearchCV is used as an additional improvement to tune the Decision Tree.

The search evaluates:

- Criterion: `gini`, `entropy`
- Maximum depth: `3`, `5`, `7`, `10`
- Minimum samples per leaf: `5`, `10`, `20`, `30`
- Class weight: `balanced`

A 5-fold `StratifiedKFold` cross-validation strategy is used with `random_state=42`.

The optimization metric is F1-score for the positive class (`Churn = Yes`).

The selected configuration was:

- Criterion: `entropy`
- Maximum depth: `7`
- Minimum samples per leaf: `30`
- Class weight: `balanced`

Mean cross-validation F1-score:

```text
0.60796
```

The tuned Decision Tree is used as the final saved model because the assignment requires a Decision Tree Classifier and the tuned model provides strong churn detection while remaining straightforward to interpret.

---

## 8. Class Imbalance

The target distribution is approximately:

- No churn: 73.46%
- Churn: 26.54%

Because the classes are imbalanced, accuracy alone is not sufficient for evaluating the model.

The Decision Tree uses `class_weight="balanced"` so that the minority churn class receives greater weight during training.

Accuracy is therefore considered together with Precision, Recall, and F1-score.

---

## 9. Additional Model Comparison

As additional analysis, the tuned Decision Tree is compared with Logistic Regression and Random Forest.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Tuned Decision Tree | 0.7402 | 0.5068 | 0.7914 | 0.6180 |
| Logistic Regression | 0.7392 | 0.5057 | 0.7897 | 0.6166 |
| Random Forest | 0.7610 | 0.5341 | 0.7825 | 0.6349 |

Random Forest performs strongly in this comparison. However, the final submission artifact remains the tuned Decision Tree because the assignment specifically requires a Decision Tree Classifier.

The comparison is included as supporting analysis rather than as a replacement for the required model.

---

## 10. Final Model Evaluation

The final tuned Decision Tree is evaluated on the held-out 30% test set.

| Metric | Score |
|---|---:|
| Accuracy | 0.7402 |
| Precision | 0.5068 |
| Recall | 0.7914 |
| F1-score | 0.6180 |

A confusion matrix is generated in the notebook.

### Business Interpretation

The model identifies approximately 79.14% of the customers who actually churn in the held-out test set.

Precision is approximately 50.68%, meaning that around half of the customers flagged as likely churners actually churn.

For a proactive retention campaign, Recall is an important metric because failing to identify a genuine churner can result in a missed retention opportunity. Some false positives may be acceptable when the cost of a retention action is lower than the cost of losing a customer.

The final business threshold or metric priority should ultimately depend on the company's retention cost and customer lifetime value.

---

## 11. Model Interpretation

Feature importance is calculated from the final Decision Tree after preprocessing.

The most important features include:

| Feature | Importance |
|---|---:|
| Contract - Month-to-month | 0.4837 |
| tenure | 0.1235 |
| InternetService - Fiber optic | 0.0849 |
| MonthlyCharges | 0.0643 |
| AvgMonthlySpend | 0.0423 |
| OnlineSecurity - No | 0.0374 |
| TotalCharges | 0.0371 |
| PaymentMethod - Electronic check | 0.0246 |
| TechSupport - No | 0.0225 |
| StreamingMovies - No internet service | 0.0135 |

The notebook also generates a visual representation of the final Decision Tree.

Feature importance indicates which features contributed most to the tree's decisions. It should not be interpreted as proof that those features independently cause churn.

---

## 12. Saved Model and Artifacts

The final trained pipeline is saved as:

```text
model/churn_model.pkl
```

The saved pipeline contains the feature engineering, preprocessing, categorical encoding, and trained Decision Tree. This allows the same transformation logic used during training to be reused by the API.

Additional artifacts include:

```text
model/decision_tree.png
model/feature_importance.csv
model/model_comparison.csv
```

### Model Pipeline

```text
Raw Customer Data
       |
       v
CustomerFeatureEngineer
       |
       v
Preprocessing
   /          \
  v            v
Numerical    Categorical
  |              |
  v              v
Imputation    Imputation
                 |
                 v
          One-Hot Encoding
   \             /
    \           /
     v         v
 DecisionTreeClassifier
          |
          v
   Churn Prediction
```

---

## 13. REST API

A Flask REST API is provided in `app.py`.

### Health Check

```text
GET /health
```

Example:

```text
http://127.0.0.1:5000/health
```

### Prediction

```text
POST /predict
```

The endpoint accepts customer information as JSON, validates the input, passes it through the saved model pipeline, and returns the predicted churn class and churn probability.

Example request:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

Example response:

```json
{
  "churn_probability": 0.7669,
  "prediction": "Yes"
}
```

The same request is available in:

```text
sample_request.json
```

Example using the sample request file:

```powershell
curl -X POST http://127.0.0.1:5000/predict `
  -H "Content-Type: application/json" `
  --data-binary "@sample_request.json"
```

The API returns HTTP 400 for invalid input and HTTP 503 if the saved model is unavailable.

---

## 14. Project Structure

```text
assignment_customer_churn/
├── data/
│   ├── TelcoCustomerChurn.csv
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── model/
│   ├── churn_model.pkl
│   ├── decision_tree.png
│   ├── feature_importance.csv
│   └── model_comparison.csv
├── notebook/
│   └── churn_analysis.ipynb
├── src/
│   └── model_utils.py
├── app.py
├── requirements.txt
├── sample_request.json
├── README.md
└── .gitignore
```

---

## 15. Setup and Execution

The project uses standard Python virtual environments. Anaconda is not required.

### Create a virtual environment

From the project directory:

```powershell
python -m venv .venv
```

### Activate the environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell execution policy prevents activation, use Command Prompt:

```cmd
.venv\Scripts\activate
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the notebook

```powershell
jupyter notebook
```

Open:

```text
notebook/churn_analysis.ipynb
```

Run the notebook from top to bottom.

The notebook performs the analysis, trains the models, evaluates the final model, and saves the model artifacts under `model/`.

### Run the API

From the project root:

```powershell
python app.py
```

The API runs on:

```text
http://127.0.0.1:5000
```

Test the health endpoint:

```powershell
curl http://127.0.0.1:5000/health
```

Then use `sample_request.json` with the `/predict` endpoint.

---

## 16. Requirements

The main dependencies are listed in `requirements.txt`.

They include:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
- ipykernel
- joblib
- Flask

Python 3.10+ is recommended.

---

## 17. Submission Contents

The submission includes:

- the complete analysis notebook;
- the dataset and data dictionary;
- reusable preprocessing and feature-engineering code;
- Decision Tree configurations and tuning;
- final trained Decision Tree model;
- model evaluation metrics and confusion matrix;
- feature importance results;
- Decision Tree visualization;
- additional model comparison;
- Flask REST API;
- sample API request;
- `requirements.txt`;
- `README.md`; and
- `.gitignore`.

The project is self-contained and does not require downloading the dataset during execution.
