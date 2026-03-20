import os
import pandas as pd
import numpy as np
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

DATASETS = {
    "511285_Oakwood_Rd_Llanedeyrn": "511285.csv",
    "511320_Abercynon_St_Westmoors": "511320.csv",
    "511469_Heol_Briwnant_South_Llanishen": "511469.csv"
}

OUTPUT_DIR = "project_results"
CLEAN_DIR = os.path.join(OUTPUT_DIR, "cleaned_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

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

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

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

def run_arima(train, test, horizon_steps):
    train_series = train["load_kw"]
    # Hardcoding order for now to test the loop
    history = list(train_series.values)
    predictions = []
    
    for i in range(len(test)):
        model = ARIMA(history, order=(1, 1, 1))
        fitted_model = model.fit()
        yhat = fitted_model.forecast(steps=horizon_steps)[-1]
        predictions.append(yhat)
        history.append(test["load_kw"].iloc[i])
        
    return np.array(predictions), (1, 1, 1)

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


def save_forecast_plot(y_true, y_pred, title, save_path):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:200], label="Actual")
    plt.plot(y_pred[:200], label="Predicted")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Load (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


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

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_test)
    return preds, model

def main():
    all_results = []
    horizons = {"10min": 1, "1hour": 6, "6hour": 36, "24hour": 144}

    lstm_y_test, lstm_pred, _ = run_lstm(
                clean_series,
                horizon_steps=horizon_steps
            )
            
    lstm_mae, lstm_rmse, lstm_mape = evaluate_forecast(lstm_y_test, lstm_pred)
    
    all_results.append({
        "Dataset": dataset_name,
        "Horizon": horizon_name,
        "Model": "LSTM",
        "MAE": lstm_mae,
        "RMSE": lstm_rmse,
        "MAPE": lstm_mape,
        "Extra": ""
    })

    for dataset_name, file_path in DATASETS.items():
        clean_series = load_and_clean_substation(file_path, dataset_name)

        for horizon_name, horizon_steps in horizons.items():
            feature_data = create_features(clean_series, horizon_steps)
            train, val, test = train_test_split_time_series(feature_data)
            y_test = test["target"].values

            # Run ARIMA
            arima_pred, _ = run_arima(train, test, horizon_steps)
            # Run XGBoost
            xgb_pred, _ = run_xgboost(train, val, test)
            
            # Save plots and log basic metrics
            print(f"Completed {dataset_name} for {horizon_name}")


def create_lstm_sequences(series, seq_length, horizon_steps):
    X, y = [], []
    values = series.values
    for i in range(len(values) - seq_length - horizon_steps):
        X.append(values[i:i+seq_length])
        y.append(values[i+seq_length+horizon_steps-1])
    return np.array(X), np.array(y)


def run_lstm(full_series, train_ratio=0.7, val_ratio=0.15, seq_length=36, horizon_steps=1):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_series[["load_kw"]])
    scaled_df = pd.DataFrame(scaled, index=full_series.index, columns=["load_kw"])
    
    X, y = create_lstm_sequences(scaled_df["load_kw"], seq_length, horizon_steps)
    
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_length, 1)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X[val_end:].reshape((X[val_end:].shape[0], X[val_end:].shape[1], 1))
    y_test = y[val_end:]

    pred_scaled = model.predict(X_test, verbose=0)
    
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    pred_actual = scaler.inverse_transform(pred_scaled).flatten()
    
    return y_test_actual, pred_actual, model



def diebold_mariano_test(y_true, pred1, pred2, h=1, power=2):
    y_true, pred1, pred2 = np.array(y_true), np.array(pred1), np.array(pred2)
    e1, e2 = y_true - pred1, y_true - pred2
    d = (e1**2) - (e2**2) if power==2 else np.abs(e1)-np.abs(e2)
    
    mean_d = np.mean(d)
    n = len(d)
    
    def autocovariance(x, lag):
        x_mean = np.mean(x)
        return np.sum((x[:n-lag] - x_mean) * (x[lag:] - x_mean)) / n

    gamma0 = autocovariance(d, 0)
    dm_stat = mean_d / np.sqrt(gamma0 / n)
    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=n - 1))
    
    variance_d = autocovariance(d, 0)
    for lag in range(1, h):
        gamma = autocovariance(d, lag)
        variance_d += 2 * gamma

    dm_stat = mean_d / np.sqrt(variance_d / n)
    harvey_adj = np.sqrt((n + 1 - 2*h + (h*(h-1)/n)) / n)
    dm_stat *= harvey_adj

    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=n - 1))
    return dm_stat, p_value