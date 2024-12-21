# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Step 1: Data Loading
data = pd.read_csv("synthetic_telecom_churn_dataset.csv")  # Ensure this file is in the same folder as this script

# Step 2: Data Exploration and Preprocessing
# Checking for missing values
if data.isnull().sum().sum() > 0:
    data.fillna(data.mean(), inplace=True)

# Encoding categorical variables (assuming 'region' is categorical)
le = LabelEncoder()
data['region'] = le.fit_transform(data['region'])

# Separating features and target variable
X = data.drop(columns=['churn'])
y = data['churn']

# Step 3: Handling Class Imbalance with SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 4: Splitting Data for Training and Testing
# Assuming 'year' column exists for year-based split
train_data = data[data['year'] < 2022]  # First two years
test_data = data[data['year'] == 2022]  # Last year

X_train = train_data.drop(columns=['churn'])
y_train = train_data['churn']
X_test = test_data.drop(columns=['churn'])
y_test = test_data['churn']

# Standardizing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Model Training and Evaluation
# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_lr))
print("Logistic Regression AUC-ROC:", roc_auc_score(y_test, y_pred_lr))

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("\nDecision Tree Results:")
print(classification_report(y_test, y_pred_dt))
print("Decision Tree AUC-ROC:", roc_auc_score(y_test, y_pred_dt))

# SGD Model
sgd_model = SGDClassifier()
sgd_model.fit(X_train, y_train)
y_pred_sgd = sgd_model.predict(X_test)
print("\nSGD (Online Learning) Results:")
print(classification_report(y_test, y_pred_sgd))
print("SGD AUC-ROC:", roc_auc_score(y_test, y_pred_sgd))

# Ensemble Model with Voting
ensemble_model = VotingClassifier(estimators=[
    ('lr', LogisticRegression()), 
    ('dt', DecisionTreeClassifier()),
    ('sgd', SGDClassifier())
], voting='soft')
ensemble_model.fit(X_train, y_train)
ensemble_pred = ensemble_model.predict(X_test)
print("\nEnsemble Model Results:")
print(classification_report(y_test, ensemble_pred))
print("Ensemble AUC-ROC:", roc_auc_score(y_test, ensemble_pred))

# Final Summary Report
print("\n===== Final Model Performance Summary =====")
print("Logistic Regression AUC-ROC:", roc_auc_score(y_test, y_pred_lr))
print("Decision Tree AUC-ROC:", roc_auc_score(y_test, y_pred_dt))
print("SGD (Online Learning) AUC-ROC:", roc_auc_score(y_test, y_pred_sgd))
print("Ensemble Model AUC-ROC:", roc_auc_score(y_test, ensemble_pred))
