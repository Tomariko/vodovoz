import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from updater import download_binance_daily_data, download_coingecko_data, download_binance_current_day_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN, CG_API_KEY

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")

def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data(token, training_days, region, data_provider):
    if data_provider == "coingecko":
        return download_data_coingecko(token, training_days)
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")

def format_data(files, data_provider):
    if not files:
        print("Already up to date")
        return
    
    price_df = pd.DataFrame()
    if data_provider == "binance":
        for file in sorted(files):
            zip_file_path = os.path.join(binance_data_path, file)

            if not zip_file_path.endswith(".zip"):
                continue

            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
            df.columns = [
                "start_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "end_time",
                "volume_usd",
                "n_trades",
                "taker_volume",
                "taker_volume_usd",
            ]
            df.index = [pd.Timestamp(x + 1, unit="ms").to_datetime64() for x in df["end_time"]]
            df.index.name = "date"
            price_df = pd.concat([price_df, df])

    elif data_provider == "coingecko":
        for file in sorted(files):
            with open(os.path.join(coingecko_data_path, file), "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df.columns = ["timestamp", "open", "high", "low", "close"]
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("date", inplace=True)
                price_df = pd.concat([price_df, df])

    price_df.sort_index(inplace=True)
    price_df.to_csv(training_price_data_path)

def load_frame(frame, timeframe):
    print(f"Loading data...")
    df = frame.loc[:, ['open', 'high', 'low', 'close']].dropna()
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
    df['date'] = frame['date'].apply(pd.to_datetime)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    return df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()

def prepare_lstm_data(df, time_step=10):
    X, y = [], []
    for i in range(len(df) - time_step):
        X.append(df[i:(i + time_step), 0])
        y.append(df[i + time_step, 0])
    return np.array(X), np.array(y)

def train_model(timeframe):
    price_data = pd.read_csv(training_price_data_path)
    df = load_frame(price_data, timeframe)
    
    df_values = df[['close']].values
    X, y = prepare_lstm_data(df_values)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    print(f"Training data shape: {X.shape}, {y.shape}")

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=50)

    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    model.save(model_file_path)

    print(f"Trained LSTM model saved to {model_file_path}")

def get_inference(token, timeframe, region, data_provider):
    loaded_model = Sequential()
    loaded_model = keras.models.load_model(model_file_path)

    if data_provider == "coingecko":
        X_new = load_frame(download_coingecko_current_day_data(token, CG_API_KEY), timeframe)
    else:
        X_new = load_frame(download_binance_current_day_data(f"{TOKEN}USDT", region), timeframe)

    X_new_values = X_new[['close']].values
    X_new_processed, _ = prepare_lstm_data(X_new_values)
    X_new_processed = X_new_processed.reshape((X_new_processed.shape[0], X_new_processed.shape[1], 1))

    current_price_pred = loaded_model.predict(X_new_processed)

    return current_price_pred[-1, 0]