import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# ==========================
# 1️⃣ Load & Prepare Data
# ==========================

def load_and_prepare_data(path):
    data = pd.read_csv(path)

    # Convert Date column
    data["Date"] = pd.to_datetime(data["Date"])
    data["Month"] = data["Date"].dt.to_period("M")

    # Group monthly spending
    monthly = data.groupby("Month")["Amount"].sum().reset_index()

    # Sort by month
    monthly = monthly.sort_values("Month").reset_index(drop=True)

    # Create numerical month index
    monthly["Month_Index"] = np.arange(len(monthly))

    # Create Lag Feature (Previous Month Spending)
    monthly["Prev_Month"] = monthly["Amount"].shift(1)

    # Rolling average (3 months)
    monthly["Rolling_3"] = monthly["Amount"].rolling(window=3).mean()

    # Remove NaN rows caused by lag/rolling
    monthly = monthly.dropna().reset_index(drop=True)

    return monthly


# ==========================
# 2️⃣ Train Model
# ==========================

def train_model(monthly):
    X = monthly[["Month_Index", "Prev_Month", "Rolling_3"]]
    y = monthly["Amount"]

    # Time-series split (no shuffle!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("📊 Model Performance")
    print("R² Score:", round(r2, 3))
    print("MAE:", round(mae, 2))

    if r2 < 0.3:
        print("⚠ Weak trend detected. Spending may be irregular.")
    elif r2 < 0.7:
        print("📈 Moderate predictive performance.")
    else:
        print("🔥 Strong predictive performance!")

    return model, X, y


# ==========================
# 3️⃣ Predict Future Months
# ==========================

def predict_future(model, monthly, months_ahead=3):
    future_predictions = []
    temp_data = monthly.copy()

    for i in range(months_ahead):
        next_index = temp_data["Month_Index"].max() + 1
        prev_month = temp_data["Amount"].iloc[-1]
        rolling_3 = temp_data["Amount"].iloc[-3:].mean()

        new_X = pd.DataFrame({
            "Month_Index": [next_index],
            "Prev_Month": [prev_month],
            "Rolling_3": [rolling_3]
        })

        prediction = model.predict(new_X)[0]
        future_predictions.append(prediction)

        # Append predicted value to temp_data for next iteration
        new_row = {
            "Month": None,
            "Amount": prediction,
            "Month_Index": next_index,
            "Prev_Month": prev_month,
            "Rolling_3": rolling_3
        }

        temp_data = pd.concat([temp_data, pd.DataFrame([new_row])], ignore_index=True)

    return future_predictions


# ==========================
# 4️⃣ Visualization
# ==========================

def plot_results(monthly, model, future_predictions):
    X_all = monthly[["Month_Index", "Prev_Month", "Rolling_3"]]
    y_all = monthly["Amount"]

    plt.figure(figsize=(10, 6))

    # Historical Data
    plt.scatter(monthly["Month_Index"], y_all, label="Historical Data")

    # Model Fit
    plt.plot(monthly["Month_Index"], model.predict(X_all),
             label="Model Fit")

    # Future Predictions
    future_indexes = np.arange(
        monthly["Month_Index"].max() + 1,
        monthly["Month_Index"].max() + 1 + len(future_predictions)
    )

    plt.scatter(future_indexes, future_predictions,
                marker='x', s=150, label="Future Predictions")

    plt.xlabel("Month Index")
    plt.ylabel("Total Spending")
    plt.title("📈 Monthly Spending Forecast")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==========================
# 5️⃣ Main Execution
# ==========================

if __name__ == "__main__":
    file_path = "expenses.csv"

    monthly_data = load_and_prepare_data(file_path)

    model, X, y = train_model(monthly_data)

    future_preds = predict_future(model, monthly_data, months_ahead=3)

    print("\n🔮 Future Predictions:")
    for i, pred in enumerate(future_preds, 1):
        print(f"Month +{i}: $ {round(pred, 2)}")

    plot_results(monthly_data, model, future_preds)