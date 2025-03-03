import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.models import Sequential

def fetch_stock_data(stock, start, end):
    data = yf.download(stock, start, end)
    data.reset_index(inplace=True)
    return data

def plot_stock_data(data):
    ma_100_days = data['Close'].rolling(100).mean()
    ma_200_days = data['Close'].rolling(200).mean()
    
    plt.figure(figsize=(10,5))
    plt.plot(ma_100_days, 'r', label="100-day MA")
    plt.plot(ma_200_days, 'b', label="200-day MA")
    plt.plot(data['Close'], 'g', label="Close Price")
    plt.legend()
    st.pyplot(plt)

def preprocess_data(data):
    data.dropna(inplace=True)
    data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
    data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])
    scaler = MinMaxScaler(feature_range=(0,1))
    data_train_scale = scaler.fit_transform(data_train)
    return data_train, data_test, data_train_scale, scaler

def prepare_train_data(data_train_scale):
    x_train, y_train = [], []
    for i in range(100, data_train_scale.shape[0]):
        x_train.append(data_train_scale[i-100:i])
        y_train.append(data_train_scale[i,0])
    return np.array(x_train), np.array(y_train)

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape, 1)),
        LSTM(units=50, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(units=60, activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(units=80, activation='relu', return_sequences=True),
        Dropout(0.4),
        LSTM(units=120, activation='relu'),
        Dropout(0.5),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    st.title("Stock Price Prediction using LSTM")
    stock = st.text_input("Enter Stock Symbol (e.g., GOOG, AAPL, TSLA):", "GOOG")
    start = '2012-01-01'
    end = '2022-12-21'
    
    if st.button("Fetch Data"):
        data = fetch_stock_data(stock, start, end)
        st.write(data.tail())
        plot_stock_data(data)
        
        data_train, data_test, data_train_scale, scaler = preprocess_data(data)
        x_train, y_train = prepare_train_data(data_train_scale)
        
        model = build_lstm_model(x_train.shape[1])
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
        
        st.success("Model Training Completed")
        model.save('stock_predictions_model.keras')
        with open("stock_predictions_model.keras", "rb") as file:
            st.download_button("Download Model", file, file_name="stock_predictions_model.keras")
        
        # Preparing test data
        past_100_days = data_train.tail(100)
        data_test = pd.concat([past_100_days, data_test], ignore_index=True)
        data_test_scale = scaler.transform(data_test)
        x_test, y_test = [], []
        for i in range(100, data_test_scale.shape[0]):
            x_test.append(data_test_scale[i-100:i])
            y_test.append(data_test_scale[i,0])
        x_test, y_test = np.array(x_test), np.array(y_test)
        
        y_predict = model.predict(x_test)
        y_predict = y_predict * scaler.data_max_
        y_test = y_test * scaler.data_max_
        
        plt.figure(figsize=(10,6))
        plt.plot(y_test, 'g', label='Actual Price')
        plt.plot(y_predict, 'r', linestyle='dashed', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()
