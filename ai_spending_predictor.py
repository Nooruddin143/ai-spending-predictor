import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load expense data
data = pd.read_csv("expenses.csv")

# Convert Date column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Create Month column
data["Month"] = data["Date"].dt.to_period("M")

# Group by month and sum spending
monthly_spending = data.groupby("Month")["Amount"].sum().reset_index()

# Convert Month to numerical index
monthly_spending["Month_Index"] = np.arange(len(monthly_spending))

# Prepare data for ML model
X = monthly_spending[["Month_Index"]]
y = monthly_spending["Amount"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Calculate R² score
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print("Model R² Score:", round(r2, 3))
# Predict next month
next_month_index = [[len(monthly_spending)]]
prediction = model.predict(next_month_index)

print("📊 Predicted Next Month Spending: $", round(prediction[0], 2))

# Plot results
plt.scatter(monthly_spending["Month_Index"], y)
plt.plot(monthly_spending["Month_Index"], model.predict(X))
plt.scatter(len(monthly_spending), prediction, marker='x', s=200)
plt.xlabel("Month Index")
plt.ylabel("Total Spending")
plt.title("AI Spending Prediction")
plt.show()