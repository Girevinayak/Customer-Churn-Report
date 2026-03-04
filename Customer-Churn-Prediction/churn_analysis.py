import pandas as pd
import numpy as np

# -------------------------------
# Create better synthetic data
# -------------------------------
np.random.seed(42)
n = 5000

data = pd.DataFrame({
    "customer_id": range(1, n+1),
    "tenure": np.random.randint(1, 72, n),
    "monthly_charges": np.random.uniform(20, 120, n),
    "contract_type": np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.6, 0.25, 0.15]
    ),
    "internet_service": np.random.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.4, 0.4, 0.2]
    )
})

# smarter churn logic (IMPORTANT)
data["churn"] = (
    (data["contract_type"] == "Month-to-month") &
    (data["monthly_charges"] > 70)
).astype(int)

# save raw data
data.to_csv("customer_churn.csv", index=False)

print("Dataset created!")

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("customer_churn.csv")

# encode
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("churn", axis=1)
y = df_encoded["churn"]

# -------------------------------
# Train model
# -------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -------------------------------
# Probability output
# -------------------------------
y_prob = model.predict_proba(X_test)[:, 1]

results = X_test.copy()
results["actual_churn"] = y_test.values
results["predicted_churn"] = y_pred
results["churn_probability"] = y_prob

results.to_csv("churn_predictions.csv", index=False)

print("Prediction file saved!")