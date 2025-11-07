# model/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv('../dataset/career_data_preprocessed.csv')

# Features & target
X = df.drop(columns=['Recommended_Career'])
y = df['Recommended_Career']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=250, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc * 100:.2f}%\n")

report = classification_report(y_test, y_pred)
print(report)

# Save report
with open("../reports/accuracy_report.txt", "w") as f:
    f.write(f"Model Accuracy: {acc*100:.2f}%\n\n")
    f.write(report)

# Confusion Matrix
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title("Career Prediction Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("../reports/confusion_matrix.png")
plt.close()

# Feature importance visualization
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(12,6))
importances[:15].plot(kind='bar')
plt.title("Top 15 Feature Importances in Career Prediction")
plt.ylabel("Importance")
plt.savefig("../model/feature_importance.png")
plt.close()

# Save model
joblib.dump(model, "career_model.pkl")
print("🎯 Model and reports saved successfully!")