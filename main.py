import os
import pandas as pd
import numpy as np
from math import sqrt
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