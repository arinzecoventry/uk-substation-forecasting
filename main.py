import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import sqrt
from scipy.stats import t
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

DATASETS = {
    "511285_Oakwood_Rd_Llanedeyrn": "511285.csv",
    "511320_Abercynon_St_Westmoors": "511320.csv",
    "511469_Heol_Briwnant_South_Llanishen": "511469.csv"
}

OUTPUT_DIR = "project_results"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
CLEAN_DIR = os.path.join(OUTPUT_DIR, "cleaned_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    return mae, rmse, mape


def diebold_mariano_test(y_true, pred1, pred2, h=1, power=2):
    y_true = np.array(y_true)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)

    e1 = y_true - pred1
    e2 = y_true - pred2

    d = (e1 ** 2) - (e2 ** 2) if power == 2 else np.abs(e1) - np.abs(e2)

    mean_d = np.mean(d)
    n = len(d)

    def autocovariance(x, lag):
        x_mean = np.mean(x)
        return np.sum((x[:n-lag] - x_mean) * (x[lag:] - x_mean)) / n

    gamma0 = autocovariance(d, 0)
    variance_d = gamma0

    max_lags = min(h - 1, int(n ** 0.25))
    for lag in range(1, max_lags + 1):
        gamma = autocovariance(d, lag)
        variance_d += 2 * gamma

    if variance_d <= 0 or np.isnan(variance_d):
        return np.nan, np.nan

    dm_stat = mean_d / np.sqrt(variance_d / n)

    harvey_inside = (n + 1 - 2*h + (h*(h-1)/n)) / n
    if harvey_inside > 0:
        dm_stat *= np.sqrt(harvey_inside)

    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=n - 1))
    return dm_stat, p_value

def load_and_clean_substation(file_path, dataset_name):
    df = pd.read_csv(file_path)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)

    df_kw = df[df["Units"] == "kW"].copy()

    series = (
        df_kw.groupby("Timestamp")["Value"]
        .sum()
        .sort_index()
        .to_frame(name="load_kw")
    )

    full_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="10min",
        tz="UTC"
    )
    series = series.reindex(full_index)

    series["load_kw"] = series["load_kw"].interpolate(method="time")

    clean_path = os.path.join(CLEAN_DIR, f"{dataset_name}_cleaned.csv")
    series.to_csv(clean_path, index_label="Timestamp")

    return series


def create_features(df, horizon_steps):
    data = df.copy()

    data["hour"] = data.index.hour
    data["dayofweek"] = data.index.dayofweek
    data["month"] = data.index.month
    data["dayofmonth"] = data.index.day
    data["weekend"] = (data.index.dayofweek >= 5).astype(int)

    data["lag_1"] = data["load_kw"].shift(1)  
    data["lag_6"] = data["load_kw"].shift(6)     
    data["lag_18"] = data["load_kw"].shift(18) 
    data["lag_36"] = data["load_kw"].shift(36)   
    data["lag_144"] = data["load_kw"].shift(144) 

    data["roll_mean_6"] = data["load_kw"].rolling(6).mean()
    data["roll_mean_18"] = data["load_kw"].rolling(18).mean()
    data["roll_mean_144"] = data["load_kw"].rolling(144).mean()

    data["target"] = data["load_kw"].shift(-horizon_steps)

    data = data.dropna()
    return data


def train_test_split_time_series(data, train_ratio=0.7, val_ratio=0.15):
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = data.iloc[:train_end].copy()
    val = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()

    return train, val, test



def find_best_arima_order(train_series):
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in [0, 1, 2, 3]:
        for d in [0, 1]:
            for q in [0, 1, 2, 3]:
                try:
                    model = ARIMA(train_series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                except:
                    continue

    return best_order, best_model


def run_arima(train, test, horizon_steps):
    train_series = train["load_kw"]
    test_target = test["target"].values

    best_order, _ = find_best_arima_order(train_series)

    history = list(train_series.values)
    predictions = []

    for i in range(len(test)):
        try:
            model = ARIMA(history, order=best_order)
            fitted_model = model.fit()

            yhat = fitted_model.forecast(steps=horizon_steps)[-1]
            predictions.append(yhat)

            history.append(test["load_kw"].iloc[i])

        except:
            predictions.append(history[-1])
            history.append(test["load_kw"].iloc[i])

    return np.array(predictions), best_order



def run_xgboost(train, val, test):
    features = [
        "lag_1", "lag_6", "lag_18", "lag_36", "lag_144",
        "roll_mean_6", "roll_mean_18", "roll_mean_144",
        "hour", "dayofweek", "month", "dayofmonth", "weekend"
    ]

    X_train = train[features]
    y_train = train["target"]
    X_val = val[features]
    y_val = val["target"]

    X_test = test[features]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict(X_test)
    return preds, model



def create_lstm_sequences(series, seq_length, horizon_steps):
    X, y = [], []
    values = series.values

    for i in range(len(values) - seq_length - horizon_steps):
        X.append(values[i:i+seq_length])
        y.append(values[i+seq_length+horizon_steps-1])

    X = np.array(X)
    y = np.array(y)

    return X, y


def run_lstm(full_series, train_ratio=0.7, val_ratio=0.15, seq_length=36, horizon_steps=1):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_series[["load_kw"]])

    scaled_df = pd.DataFrame(
        scaled,
        index=full_series.index,
        columns=["load_kw"]
    )

    X, y = create_lstm_sequences(scaled_df["load_kw"], seq_length, horizon_steps)

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_length, 1)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    pred_scaled = model.predict(X_test, verbose=0)

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    pred_actual = scaler.inverse_transform(pred_scaled).flatten()

    return y_test_actual, pred_actual, model


def save_forecast_plot(y_true, y_pred, title, save_path):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:200], label="Actual")
    plt.plot(y_pred[:200], label="Predicted")
    plt.title(title)
    plt.xlabel("Time Step (10 mins interval)")
    plt.ylabel("Load (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def main():
    all_results = []
    dm_results = []

    horizons = {
        "10min": 1,
        "1hour": 6,
        "6hour": 36,
        "24hour": 144
    }

    for dataset_name, file_path in DATASETS.items():
        print(f"\nProcessing {dataset_name} ...")

        clean_series = load_and_clean_substation(file_path, dataset_name)

        for horizon_name, horizon_steps in horizons.items():
            print(f"  Horizon: {horizon_name}")

            feature_data = create_features(clean_series, horizon_steps)
            train, val, test = train_test_split_time_series(feature_data)

            y_test = test["target"].values

            arima_pred, best_order = run_arima(train, test, horizon_steps)
            arima_mae, arima_rmse, arima_mape = evaluate_forecast(y_test, arima_pred)

            save_forecast_plot(
                y_test,
                arima_pred,
                f"{dataset_name} - ARIMA - {horizon_name}",
                os.path.join(PLOT_DIR, f"{dataset_name}_ARIMA_{horizon_name}.png")
            )

            all_results.append({
                "Dataset": dataset_name,
                "Horizon": horizon_name,
                "Model": "ARIMA",
                "MAE": arima_mae,
                "RMSE": arima_rmse,
                "MAPE": arima_mape,
                "Extra": str(best_order)
            })

            xgb_pred, _ = run_xgboost(train, val, test)
            xgb_mae, xgb_rmse, xgb_mape = evaluate_forecast(y_test, xgb_pred)

            save_forecast_plot(
                y_test,
                xgb_pred,
                f"{dataset_name} - XGBoost - {horizon_name}",
                os.path.join(PLOT_DIR, f"{dataset_name}_XGBoost_{horizon_name}.png")
            )

            all_results.append({
                "Dataset": dataset_name,
                "Horizon": horizon_name,
                "Model": "XGBoost",
                "MAE": xgb_mae,
                "RMSE": xgb_rmse,
                "MAPE": xgb_mape,
                "Extra": ""
            })

            lstm_y_test, lstm_pred, _ = run_lstm(
                clean_series,
                train_ratio=0.7,
                val_ratio=0.15,
                seq_length=36,
                horizon_steps=horizon_steps
            )

            lstm_mae, lstm_rmse, lstm_mape = evaluate_forecast(lstm_y_test, lstm_pred)

            save_forecast_plot(
                lstm_y_test,
                lstm_pred,
                f"{dataset_name} - LSTM - {horizon_name}",
                os.path.join(PLOT_DIR, f"{dataset_name}_LSTM_{horizon_name}.png")
            )

            all_results.append({
                "Dataset": dataset_name,
                "Horizon": horizon_name,
                "Model": "LSTM",
                "MAE": lstm_mae,
                "RMSE": lstm_rmse,
                "MAPE": lstm_mape,
                "Extra": ""
            })

            min_len = min(len(y_test), len(arima_pred), len(xgb_pred), len(lstm_y_test), len(lstm_pred))

            y_dm = y_test[-min_len:]
            arima_dm = arima_pred[-min_len:]
            xgb_dm = xgb_pred[-min_len:]

            y_lstm_dm = lstm_y_test[-min_len:]
            lstm_dm = lstm_pred[-min_len:]

            dm_ax_stat, dm_ax_p = diebold_mariano_test(y_dm, arima_dm, xgb_dm, h=horizon_steps, power=2)
            dm_al_stat, dm_al_p = diebold_mariano_test(y_lstm_dm, arima_dm, lstm_dm, h=horizon_steps, power=2)
            dm_xl_stat, dm_xl_p = diebold_mariano_test(y_lstm_dm, xgb_dm, lstm_dm, h=horizon_steps, power=2)

            dm_results.append({
                "Dataset": dataset_name,
                "Horizon": horizon_name,
                "Comparison": "ARIMA vs XGBoost",
                "DM_Statistic": dm_ax_stat,
                "p_value": dm_ax_p
            })
            dm_results.append({
                "Dataset": dataset_name,
                "Horizon": horizon_name,
                "Comparison": "ARIMA vs LSTM",
                "DM_Statistic": dm_al_stat,
                "p_value": dm_al_p
            })
            dm_results.append({
                "Dataset": dataset_name,
                "Horizon": horizon_name,
                "Comparison": "XGBoost vs LSTM",
                "DM_Statistic": dm_xl_stat,
                "p_value": dm_xl_p
            })

    results_df = pd.DataFrame(all_results)
    dm_df = pd.DataFrame(dm_results)

    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results.csv"), index=False)
    dm_df.to_csv(os.path.join(OUTPUT_DIR, "dm_test_results.csv"), index=False)

    print("\nDone.")
    print("\nSaved:")
    print("- model_results.csv")
    print("- dm_test_results.csv")
    print("- cleaned substation files")
    print("- forecast plots")


if __name__ == "__main__":
    main()