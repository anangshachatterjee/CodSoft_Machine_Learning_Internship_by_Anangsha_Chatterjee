import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import ( confusion_matrix,
                              accuracy_score,
                              precision_score,
                              recall_score,
                              f1_score,
                              classification_report)
import zipfile

# ==========================================
# Load the Datasets and Check Missing Values
# ==========================================
with zipfile.ZipFile(r"C:\Users\ANANGSHA\Downloads\archive (Customer Churn Dataset).zip") as z:

    df = pd.read_csv(z.open("Churn_Modelling.csv"))

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

# ==========================================
# Label Encoding
# ==========================================
label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])


# ==========================================
# Drop unnecessary columns
# ==========================================
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

# ==========================================
# Features and Target
# ==========================================
x = df.drop("Exited", axis=1)
y = df["Exited"]

# ==========================================
# Train Test Splitting
# ==========================================
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=40)

# ==========================================
# Train Test Splitting
# ==========================================
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ==========================================
# Logistic Regression training and predicting
# ==========================================
model_LR = LogisticRegression(max_iter=1000)
model_LR.fit(x_train_scaled, y_train)
y_pred_LR = model_LR.predict(x_test_scaled)

# ==========================================
# Metrics for Logisitic Regression
# ==========================================
print("Accuracy Logisitic Regression :", accuracy_score(y_test, y_pred_LR))
print("Precision Logisitic Regression:", precision_score(y_test, y_pred_LR))
print("Recall Logisitic Regression:", recall_score(y_test, y_pred_LR))
print("F1 Score Logisitic Regression:", f1_score(y_test, y_pred_LR))
print("Confusion Matrix Logisitic Regression:", confusion_matrix(y_test, y_pred_LR))
print("Classification Report Logistic Regression:", classification_report(y_test, y_pred_LR))

# ==========================================
# Random Forest Classifier Model Training and Predicting
# ==========================================
model_RF = RandomForestClassifier(
    n_estimators=100,
    random_state=40,
    n_jobs=1
)
model_RF.fit(x_train, y_train)
y_pred_RF = model_RF.predict(x_test)

# ==========================================
# Metrics for Random Forest Classifier
# ==========================================
print("Accuracy Random Forest Classifier :", accuracy_score(y_test, y_pred_RF))
print("Precision Random Forest Classifier:", precision_score(y_test, y_pred_RF))
print("Recall Random Forest Classifier:", recall_score(y_test, y_pred_RF))
print("F1 Score Random Forest Classifier:", f1_score(y_test, y_pred_RF))
print("Confusion Matrix Random Forest Classifier:", confusion_matrix(y_test, y_pred_RF))
print("Classification Report Random Forest Classifier:", classification_report(y_test, y_pred_RF))

# ==========================================
# Gradient Booster Classifier Model Training and Predicting
# ==========================================
model_GB = GradientBoostingClassifier(random_state=40)
model_GB.fit(x_train, y_train)
y_pred_GB = model_GB.predict(x_test)

# ==========================================
# Metrics for Gradient Booster Classifier
# ==========================================
print("Accuracy Gradient Booster Classifier :", accuracy_score(y_test, y_pred_GB))
print("Precision Gradient Booster Classifier:", precision_score(y_test, y_pred_GB))
print("Recall Gradient Booster Classifier:", recall_score(y_test, y_pred_GB))
print("F1 Score Gradient Booster Classifier:", f1_score(y_test, y_pred_GB))
print("Confusion Matrix Gradient Booster Classifier:", confusion_matrix(y_test, y_pred_GB))
print("Classification Report Gradient Booster Classifier:", classification_report(y_test, y_pred_GB))

# ==========================================
# A comparision table which will show all the metrics evaluation performed for each algorithm
# ==========================================

results = pd.DataFrame({
    "Algorithm": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_LR),
        accuracy_score(y_test, y_pred_RF),
        accuracy_score(y_test, y_pred_GB)
    ],
    "Precision": [
        precision_score(y_test, y_pred_LR),
        precision_score(y_test, y_pred_RF),
        precision_score(y_test, y_pred_GB)
    ],
    "Recall": [
        recall_score(y_test, y_pred_LR),
        recall_score(y_test, y_pred_RF),
        recall_score(y_test, y_pred_GB)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_LR),
        f1_score(y_test, y_pred_RF),
        f1_score(y_test, y_pred_GB)
    ]
})
results.iloc[:,1:] = results.iloc[:,1:].round()
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

