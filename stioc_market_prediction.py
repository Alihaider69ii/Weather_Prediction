
!#pip install yfinance tensorflow scikit-learn matplotlib --quiet

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf

print('TensorFlow version:', tf.__version__)


TICKER = 'AAPL'
START = '2015-01-01'
END = '2025-01-01'
SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 25
MODEL_SAVE_PATH = 'lstm_stock_model.h5'


df = yf.download(TICKER, start=START, end=END, progress=False)

if df.empty:
    raise RuntimeError(f"No data downloaded for {TICKER}. Check ticker symbol or date range.")

df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
print(df.tail())


plt.figure(figsize=(12, 5))
plt.plot(df['Close'], label='Close Price')
plt.title(f'{TICKER} Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


print(df.describe())


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(df[['Close']].values)


X = []
y = []
for i in range(SEQUENCE_LENGTH, len(scaled_close)):
    X.append(scaled_close[i - SEQUENCE_LENGTH:i, 0])
    y.append(scaled_close[i, 0])

X = np.array(X)
y = np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print('X shape:', X.shape)
print('y shape:', y.shape)


train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print('Train samples:', X_train.shape[0])
print('Test samples:', X_test.shape[0])


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
model.summary()


history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    shuffle=False
)


model.save(MODEL_SAVE_PATH)
print('Model saved to', MODEL_SAVE_PATH)


plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

preds = model.predict(X_test)
preds_rescaled = scaler.inverse_transform(preds.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

-
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test_rescaled, preds_rescaled)
mae = mean_absolute_error(y_test_rescaled, preds_rescaled)
rmse = np.sqrt(mse)

print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')


plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test_rescaled)), y_test_rescaled, label='Actual')
plt.plot(range(len(preds_rescaled)), preds_rescaled, label='Predicted')
plt.title(f'{TICKER} Actual vs Predicted Close Price (Test Set)')
plt.xlabel('Time steps')
plt.ylabel('Price')
plt.legend()
plt.show()


last_sequence = scaled_close[-SEQUENCE_LENGTH:]
input_seq = last_sequence.reshape(1, SEQUENCE_LENGTH, 1)
next_pred_scaled = model.predict(input_seq)
next_pred = scaler.inverse_transform(next_pred_scaled.reshape(-1, 1))[0, 0]

print(f'Predicted next close price for {TICKER}: {next_pred:.2f}')
