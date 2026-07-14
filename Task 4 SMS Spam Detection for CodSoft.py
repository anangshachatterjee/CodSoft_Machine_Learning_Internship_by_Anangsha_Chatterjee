import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ( confusion_matrix,
                              accuracy_score,
                              precision_score,
                              recall_score,
                              f1_score,
                              classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# ==========================================
# Load the Datasets and Check Missing Values
# ==========================================
with zipfile.ZipFile(r"C:\Users\ANANGSHA\Downloads\archive (SMS Spam Dataset).zip") as z:
    with z.open("spam.csv") as file:
        df = pd.read_csv(file, encoding="latin-1")

df = df[['v1', 'v2']]
df.columns = ['Label', 'Message']
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# ==========================================
# Label encoding
# ==========================================
encoder = LabelEncoder()
df['Label'] = encoder.fit_transform(df['Label'])

# ==========================================
# Features and Target
# ==========================================
x = df['Message']
y = df['Label']

# ==========================================
# Train and Test Splitting
# ==========================================
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=40, stratify=y)

# ==========================================
# Convert text into TF-IDF features
# ==========================================
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)


# ==========================================
# Logistic Regression model training
# ==========================================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(x_train_vectorized, y_train)
y_pred_lr = lr_model.predict(x_test_vectorized)

# ==========================================
# Logistic Regression model metrics evaluation
# ==========================================
print("Accuracy Logisitic Regression :", accuracy_score(y_test, y_pred_lr))
print("Precision Logisitic Regression:", precision_score(y_test, y_pred_lr))
print("Recall Logisitic Regression:", recall_score(y_test, y_pred_lr))
print("F1 Score Logisitic Regression:", f1_score(y_test, y_pred_lr))
print("Confusion Matrix Logisitic Regression:", confusion_matrix(y_test, y_pred_lr))
print("Classification Report Logistic Regression:", classification_report(y_test, y_pred_lr))

# ==========================================
# Multinomial Naive Bayes model training
# ==========================================
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(x_train_vectorized, y_train)
y_pred_nb = nb_model.predict(x_test_vectorized)

# ==========================================
# Multinomial Naive Bayes metrics evaluation
# ==========================================
print("Accuracy Multinomial Naive Bayes :", accuracy_score(y_test, y_pred_nb))
print("Precision Multinomial Naive Bayes:", precision_score(y_test, y_pred_nb))
print("Recall Multinomial Naive Bayes:", recall_score(y_test, y_pred_nb))
print("F1 Score Multinomial Naive Bayes:", f1_score(y_test, y_pred_nb))
print("Confusion Matrix Multinomial Naive Bayes:", confusion_matrix(y_test, y_pred_nb))
print("Classification Report Multinomial Naive Bayes:", classification_report(y_test, y_pred_nb))

# ==========================================
# SVM model training
# ==========================================
svm_model = LinearSVC(C=1.0, random_state=40)
svm_model.fit(x_train_vectorized, y_train)
y_pred_svm = svm_model.predict(x_test_vectorized)

# ==========================================
# SVM metrics evaluation
# ==========================================
print("Accuracy SVM :", accuracy_score(y_test, y_pred_svm))
print("Precision SVM:", precision_score(y_test, y_pred_svm))
print("Recall SVM:", recall_score(y_test, y_pred_svm))
print("F1 Score SVM:", f1_score(y_test, y_pred_svm))
print("Confusion Matrix SVM:", confusion_matrix(y_test, y_pred_svm))
print("Classification Report SVM:", classification_report(y_test, y_pred_svm))


# ==========================================
# A comparision table which will show all the metrics evaluation performed for each algorithm
# ==========================================

results = pd.DataFrame({
    "Algorithm": [
        "Logistic Regression",
        "Naive Bayes",
        "Support Vector Machine"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_svm)
    ],
    "Precision": [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_nb),
        precision_score(y_test, y_pred_svm)
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_nb),
        recall_score(y_test, y_pred_svm)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_nb),
        f1_score(y_test, y_pred_svm)
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

