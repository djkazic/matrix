from dotenv import load_dotenv, find_dotenv
import argparse
import gc
import numpy as np
import tensorflow as tf
import json
import multiprocessing
import os
import schedule
import time
from keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam

import pandas as pd
import calendar
from datetime import datetime
import bitcoinrpc
from bitcoinrpc.authproxy import AuthServiceProxy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dotenv_file = find_dotenv(usecwd=True)
load_dotenv(dotenv_file)
print(f"Env file loaded {dotenv_file}")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
rpc_host = os.getenv("RPC_HOST")
rpc_user = os.getenv("RPC_USER")
rpc_password = os.getenv("RPC_PASSWORD")
rpc_port = os.getenv("RPC_PORT")
model_path = os.getenv("MODEL_PATH")
prediction_path = os.getenv("PREDICTION_PATH")
print("All env vars loaded")

def get_current_block_hash():
    rpc_connection = AuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")
    return rpc_connection.getbestblockhash()

def calculate_purge_fee_rate():
    rpc_connection = AuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")
    mempool_info = rpc_connection.getmempoolinfo()
    mempool_size_bytes = mempool_info["bytes"]
    mempool_transactions = mempool_info["size"]
    # Calculate the fee rate (fee per byte) for the 1GB mempool
    one_gb_mempool_size_bytes = 1024 * 1024 * 1024  # Convert 1GB to bytes
    one_gb_fee_rate = mempool_transactions / one_gb_mempool_size_bytes
    # Scale the fee rate for the 300MB mempool
    scaled_fee_rate = one_gb_fee_rate * (300 * 1024 * 1024) / one_gb_mempool_size_bytes
    return scaled_fee_rate * 100000000

def job():
    # Calculate purge_fee_rate
    purge_fee_rate = calculate_purge_fee_rate() * 1.9
    print(f"Purge feeRate (thousands) = {purge_fee_rate}, setting floor")
    # Load the CSV
    df = pd.read_csv('training_data.csv', names=['date', 'block_diff', 'fee_rate'])
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M:%S")
    df['day_of_week'] = df['date'].dt.dayofweek
    df['hour_of_day'] = df['date'].dt.hour
    # Drop the original date column
    df = df.drop(columns='date')
    # Split data into X (inputs) and y (output)
    X = df[['day_of_week', 'hour_of_day', 'block_diff']]
    X_columns = ['day_of_week', 'hour_of_day', 'block_diff']
    y = df['fee_rate']
    # Split data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))
    X_train, X_test, y_train, y_test = X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
    # Existing model: load
    model_usable = False
    if os.path.exists(model_path):
        print("Loading existing model from disk")
        model = load_model(model_path)
        model_usable = True
    # TODO: model evaluation pre-load
    if not model_usable:
        print(f"No suitable model found... Training new model.")
        # Now you can use X_train, y_train, X_test, and y_test to train and test your Keras model
        # Define the model
        #initial_learning_rate = 0.009
        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #    initial_learning_rate,
        #    decay_steps=15000,
        #    decay_rate=0.96,
        #    staircase=True)
        optimizer = tf.keras.optimizers.Adam(0.0035)
        model = Sequential([
            Dense(128, input_shape=(3,), activation=LeakyReLU(alpha=0.05)),
            Dense(64, activation=LeakyReLU(alpha=0.05)),
            Dense(32, activation=LeakyReLU(alpha=0.05)),
            Dense(1, activation=LeakyReLU(alpha=0.05))
        ])

        # Compile the model
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=14, restore_best_weights=True)
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=18, shuffle=True, callbacks=[early_stop], batch_size=256)
    print(model.summary())
    validation_loss_threshold = 6.5e-05
    validation_loss = model.history.history['val_loss'][-1]
    if validation_loss > validation_loss_threshold:
        print(f"Model did not meet validation_loss requirements. Expected < {validation_loss_threshold}, actual {validation_loss} Aborting")
        return
    # Eval
    current_day = datetime.now().weekday()
    current_hour = datetime.now().hour

    # Get latest block
    current_block_hash = get_current_block_hash()

    # Do predictions
    block_target = (num for num in range(2, 43))
    fee_by_block_target = {}

    for target in block_target:
        # use the model to predict the output
        new_data_df = pd.DataFrame({'day_of_week': [current_day], 'hour_of_day': [current_hour], 'block_diff': [target]})

        # Convert DataFrame to a NumPy array
        new_data_array = new_data_df.to_numpy()

        # Scale the new_data using the scaler for input features
        new_data_scaled = scaler.transform(new_data_array)

        # use the model to predict the output
        predicted_fee_rate = model.predict(new_data_scaled)
        predicted_fee_rate = y_scaler.inverse_transform(predicted_fee_rate)
        if predicted_fee_rate < 1:
            predicted_fee_rate = 1
        # Convert to thousands
        predicted_fee_rate = int(predicted_fee_rate * 1000)
        if target > 2:
            # Check prev value > curr value
            predicted_fee_rate = min(predicted_fee_rate, int(fee_by_block_target[str(target - 1)]) * 0.9)
        if target >= 3 and target < 16:
            # Predicted fee_rate should be either the prediction or the prior target * 0.85
            predicted_fee_rate = max(predicted_fee_rate, int(fee_by_block_target[str(target - 1)] * 0.85))
        elif target >= 16:
            # prediction or prior target * 0.8
            predicted_fee_rate = max(predicted_fee_rate, int(fee_by_block_target[str(target - 1)] * 0.8))
        #if target <= 72:
        #    predicted_fee_rate = max(predicted_fee_rate, int(purge_fee_rate * 2))
        else:
            predicted_fee_rate = max(predicted_fee_rate, int(purge_fee_rate))
        predicted_fee_rate = max(predicted_fee_rate, int(purge_fee_rate))
        predicted_fee_rate = int(predicted_fee_rate)
        fee_by_block_target[str(target)] = predicted_fee_rate
        print(f"{target} target feeRate (thousands) = {predicted_fee_rate}")

    output = {
        "current_block_hash": current_block_hash,
        "fee_by_block_target": fee_by_block_target
    }
    with open(prediction_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(output)
    model.save(model_path)
    print("Saved model to disk!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Run the job immediately")
    args = parser.parse_args()
    schedule.every().hour.at(":00").do(job)
    schedule.every().hour.at(":10").do(job)
    schedule.every().hour.at(":20").do(job)
    schedule.every().hour.at(":30").do(job)
    schedule.every().hour.at(":40").do(job)
    schedule.every().hour.at(":50").do(job)
    if args.force:
        print("Force started")
        job()
    while True:
        schedule.run_pending()
        time.sleep(1)
