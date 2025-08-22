# Student Performance Predictor
# Author: Aditya Kumar Jha
# Description: Predicts Pass/Fail based on study hours, attendance, and internal marks.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


np.random.seed(42)
data = {
    "Study_Hours": np.random.randint(1, 10, 50),         
    "Attendance": np.random.randint(50, 100, 50),        
    "Internal_Marks": np.random.randint(20, 50, 50),     
}

df = pd.DataFrame(data)


df["Result"] = np.where(
    (df["Study_Hours"] * 2 + df["Attendance"] * 0.5 + df["Internal_Marks"]) > 100, 
    1, 0
)


X = df[["Study_Hours", "Attendance", "Internal_Marks"]].values
y = df["Result"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Student Performance Predictor")
print("Model Accuracy:", round(accuracy * 100, 2), "%")


print("\n--- Try Your Own Data ---")
study_hours = int(input("Enter daily study hours (1-10): "))
attendance = int(input("Enter attendance percentage (50-100): "))
internal_marks = int(input("Enter internal marks (20-50): "))

prediction = model.predict([[study_hours, attendance, internal_marks]])[0]
if prediction == 1:
    print("🎉 Prediction: PASS")
else:
    print("⚠️ Prediction: FAIL")
