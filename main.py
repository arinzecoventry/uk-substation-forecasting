import os
import pandas as pd
import numpy as np
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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