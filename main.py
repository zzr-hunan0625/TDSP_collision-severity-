"""
File: main.py
Description:
This script demonstrates the preprocessing, modeling, and evaluation of incident severity prediction 
using machine learning models, including K-Nearest Neighbors (KNN), Support Vector Machines (SVM), 
Random Forest, and Decision Tree. 

The workflow includes:
1. Loading and preprocessing data from a CSV file.
2. Standardizing numerical features for model input.
3. Handling imbalanced data using SMOTE for oversampling.
4. Training and cross-validating multiple models.
5. Evaluating the best-performing model (Random Forest) using a classification report and confusion matrix.
6. Visualizing model performance through box plots, confusion matrices, and classification metrics.

Dependencies:
- pandas
- numpy
- matplotlib
- scikit-learn
- imbalanced-learn (SMOTE)
- os

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# Get the current working directory and construct the path to the CSV file
current_working_directory = os.getcwd()
location = os.path.join(current_working_directory, "processed_data.csv")

# Load the processed data
df = pd.read_csv(location)

# Standardize numerical features for consistent scaling
scaler = StandardScaler()
df['POSTED_SPE_scaled'] = scaler.fit_transform(df[['POSTED_SPE']])
df['StreetWidt_scaled'] = scaler.fit_transform(df[['StreetWidt']])
df['Number_Tra_scaled'] = scaler.fit_transform(df[['Number_Tra']])
df['Number_Par_scaled'] = scaler.fit_transform(df[['Number_Par']])

# Select features for model training and convert data types
selected_columns = ['precipitation_sum', 'is_inalchol', 'is_inunsafespeed', 'is_any_inbike', 
                    'POSTED_SPE_scaled', 'StreetWidt_scaled', 'Number_Tra_scaled', 'Number_Par_scaled']
X = df[selected_columns].apply(pd.to_numeric, errors='coerce')
y = df['incident_severity'].apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)  # Fill missing values with 0
y.fillna(0, inplace=True)

# Handle imbalanced data using SMOTE (Synthetic Minority Oversampling Technique)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define the models to evaluate
knn = KNeighborsClassifier(n_neighbors=5)          # KNN
svm = SVC(kernel='linear', random_state=42)        # SVM
rf_model = RandomForestClassifier(random_state=42) # Random Forest
dt_model = DecisionTreeClassifier(random_state=42) # Decision Tree

models = {
    "KNN": knn,
    "SVM": svm,
    "Random Forest": rf_model,
    "Decision Tree": dt_model
}

# Perform cross-validation for each model and store results
cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores[name] = scores
    print(f"{name} Cross-validation scores: {scores}")
    print(f"{name} Mean accuracy: {np.mean(scores):.4f}")

# Train and evaluate the Random Forest model
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
classification_report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print("Random Forest Classification Report:")
print(classification_report_rf)
print("Confusion Matrix:")
print(confusion_matrix_rf)

# Visualize cross-validation results using a box plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(cv_scores.values(), labels=cv_scores.keys(), patch_artist=True)
ax.set_title("Cross-validation Accuracy Scores")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Models")
plt.show()

# Plot the confusion matrix for Random Forest
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf,
                              display_labels=["Level 1", "Level 2", "Level 3"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Visualize classification metrics (Precision, Recall, F1-Score) as bar plots
labels = list(classification_report_rf.keys())[:-3]
precision = [classification_report_rf[label]["precision"] for label in labels]
recall = [classification_report_rf[label]["recall"] for label in labels]
f1_score = [classification_report_rf[label]["f1-score"] for label in labels]
x = np.arange(len(labels))  # Indices for categories
bar_width = 0.25  # Width of each bar

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - bar_width, precision, bar_width, label="Precision")
ax.bar(x, recall, bar_width, label="Recall")
ax.bar(x + bar_width, f1_score, bar_width, label="F1-Score")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Score")
ax.set_title("Random Forest Classification Report")
ax.legend()
plt.show()





