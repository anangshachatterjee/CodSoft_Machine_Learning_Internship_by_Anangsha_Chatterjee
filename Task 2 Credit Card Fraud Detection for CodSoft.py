import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             classification_report)
import zipfile
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# Load the Datasets and Check Missing Values
# ==========================================
with zipfile.ZipFile(r"C:\Users\ANANGSHA\Downloads\archive (Credit Card Fraud Dataset).zip") as z:

    train_df = pd.read_csv(z.open("fraudTrain.csv"))
    test_df = pd.read_csv(z.open("fraudTest.csv"))

print("Training Data Shape =", train_df.shape)
print("Testing Data Shape =", test_df.shape)

print(train_df.isnull().sum())
print(test_df.isnull().sum())


# ==========================================
# Drop unnecessary columns
# ==========================================
drop_cols = [
    "Unnamed: 0",
    "trans_num",
    "first",
    "last",
    "gender",
    "street"
]
train_df.drop(columns=drop_cols, axis=1, inplace=True)
test_df.drop(columns=drop_cols, axis=1, inplace=True)

# ==========================================
# Convert into Date and Time
# ==========================================
train_df["trans_date_trans_time"] = pd.to_datetime(train_df["trans_date_trans_time"])
test_df["trans_date_trans_time"] = pd.to_datetime(test_df["trans_date_trans_time"])


for df in [train_df, test_df]:
    df["year"] = df["trans_date_trans_time"].dt.year
    df["month"] = df["trans_date_trans_time"].dt.month
    df["day"] = df["trans_date_trans_time"].dt.day
    df["hour"] = df["trans_date_trans_time"].dt.hour

    df.drop("trans_date_trans_time", axis=1, inplace=True)

for df in [train_df, test_df]:
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = 2026 - df["dob"].dt.year
    df.drop("dob", axis=1, inplace=True)

# ==========================================
# Encode Categorical Columns
# ==========================================
categorical_cols = train_df.select_dtypes(include="object").columns
encoders = {}
for col in categorical_cols:
    label_encoder = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    label_encoder.fit(combined)
    train_df[col] = label_encoder.transform(train_df[col].astype(str))
    test_df[col] = label_encoder.transform(test_df[col].astype(str))
    encoders[col] = label_encoder

# ==========================================
# Features and Target
# ==========================================
x_train = train_df.drop("is_fraud", axis=1)
y_train = train_df["is_fraud"]

x_test = test_df.drop("is_fraud", axis=1)
y_test = test_df["is_fraud"]

# ==========================================
# Data Scaling
# ==========================================
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ==========================================
# Logisitic Regression Model Training
# ==========================================
LR_model = LogisticRegression(max_iter=1000)
LR_model.fit(x_train_scaled, y_train)

# ================================================================
# Prediction of data based on test values for Logistic Regression
# ================================================================
y_pred_LR = LR_model.predict(x_test_scaled)
print(y_pred_LR[ :10])

# ================================================================
# Compute metrics for Logistic Regression
# ================================================================
accuracy_LR = accuracy_score(y_test, y_pred_LR)
print("Accuracy Logistic Regression =", accuracy_LR)

precision_LR = precision_score(y_test, y_pred_LR)
print("Precision Logistic Regression =", precision_LR)

recall_LR = recall_score(y_test, y_pred_LR)
print("Recall Logistic Regression =", recall_LR)

f1_LR = f1_score(y_test, y_pred_LR)
print("F1 Logistic Regression =", f1_LR)

cm_LR = confusion_matrix(y_test, y_pred_LR)
print("Confusion Matrix Logistic Regression =", cm_LR)

report_LR = classification_report(y_test, y_pred_LR)
print("Classification Report Logistic Regression =", report_LR)


# ==========================================
# Decision Tree Classifier Model Training
# ==========================================
DT_model = DecisionTreeClassifier(criterion = 'gini',
                               max_depth = 3,
                               min_samples_split = 2,
                               min_samples_leaf = 2,
                               max_leaf_nodes = 10,
                               random_state = 30)
DT_model.fit(x_train, y_train)

# ==========================================================================
# Prediction of data based on test values for Decision Tree Classification
# ==========================================================================
y_pred_DT = DT_model.predict(x_test)
print(y_pred_DT[ :10])

# ================================================================
# Compute metrics for Decision Tree Classification
# ================================================================
accuracy_DT = accuracy_score(y_test, y_pred_DT)
print("Accuracy Decision Tree Classification =", accuracy_DT)

precision_DT = precision_score(y_test, y_pred_DT)
print("Precision Decision Tree Classification =", precision_DT)

recall_DT = recall_score(y_test, y_pred_DT)
print("Recall Decision Tree Classification =", recall_DT)

f1_DT = f1_score(y_test, y_pred_DT)
print("F1 Decision Tree Classification =", f1_DT)

cm_DT = confusion_matrix(y_test, y_pred_DT)
print("Confusion Matrix Decision Tree Classification =", cm_DT)

report_DT = classification_report(y_test, y_pred_DT)
print("Classification Report Decision Tree Classification =", report_DT)

# ==========================================
# Random Forest Classifier Model Training
# ==========================================
RF_model = RandomForestClassifier(
    n_estimators=100,
    random_state=30,
    n_jobs=1
)

RF_model.fit(x_train, y_train)

# ==========================================================================
# Prediction of data based on test values for Random Forest Classification
# ==========================================================================
y_pred_RF = RF_model.predict(x_test)
print(y_pred_RF[ :10])

# ================================================================
# Compute metrics for Random Forest Classification
# ================================================================
accuracy_RF = accuracy_score(y_test, y_pred_RF)
print("Accuracy Random Forest Classification =", accuracy_RF)

precision_RF = precision_score(y_test, y_pred_RF)
print("Precision Random Forest Classification =", precision_RF)

recall_RF = recall_score(y_test, y_pred_RF)
print("Recall Random Forest Classification =", recall_RF)

f1_RF = f1_score(y_test, y_pred_RF)
print("F1 Random Forest Classification =", f1_RF)

cm_RF = confusion_matrix(y_test, y_pred_RF)
print("Confusion Matrix Random Forest Classification =", cm_RF)

report_RF = classification_report(y_test, y_pred_RF)
print("Classification Report Random Forest Classification =", report_RF)

# ==============================================================================================
# A comparision table which will show all the metrics evaluation performed for each algorithm
# ==============================================================================================
results = pd.DataFrame({
    "Algorithm": [
        "Logistic Regression",
        "Decision Trees",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_LR,
        accuracy_DT,
        accuracy_RF
    ],
    "Precision": [
        precision_LR,
        precision_DT,
        precision_RF
    ],
    "Recall": [
        recall_LR,
        recall_DT,
        recall_RF
    ],
    "F1 Score": [
        f1_LR,
        f1_DT,
        f1_RF
    ]
})
results.iloc[:,1:] = results.iloc[:,1:].round(2)
fig, ax = plt.subplots(figsize=(8,3))
ax.axis('off')
table = ax.table(
    cellText=results.values,
    colLabels=results.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.5, 2.0)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('lightgray')

plt.title("A comparision table which will show all the metrics evaluation performed for each algorithm",
          fontsize=14,
          fontweight='bold',
          pad=20)

plt.show()
