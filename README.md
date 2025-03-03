# Stock Price Prediction using LSTM

## Overview
This project is a Streamlit web application that predicts stock prices using a Long Short-Term Memory (LSTM) neural network. The app allows users to fetch stock data, visualize moving averages, train an LSTM model, and generate stock price predictions.

## Features
- Fetch stock market data using Yahoo Finance (`yfinance`)
- Display moving averages (100-day and 200-day) along with the closing price
- Preprocess data using `MinMaxScaler`
- Train an LSTM model using TensorFlow/Keras
- Predict future stock prices based on historical data
- Download the trained model for further use

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow/Keras
- Yahoo Finance API

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## Usage
1. Enter a stock symbol (e.g., `GOOG`, `AAPL`, `TSLA`).
2. Click the "Fetch Data" button to retrieve stock market data.
3. View stock price history and moving averages.
4. Train the LSTM model on the dataset.
5. Download the trained model after training is complete.
6. View actual vs. predicted stock prices.

## Model Architecture
- **Input Layer**: Takes in a sequence of past 100 days of stock prices.
- **LSTM Layers**:
  - 50 units (return sequences = True, dropout 0.2)
  - 60 units (return sequences = True, dropout 0.3)
  - 80 units (return sequences = True, dropout 0.4)
  - 120 units (dropout 0.5)
- **Dense Layer**: Outputs a single predicted stock price.

## Future Enhancements
- Add more technical indicators for better predictions.
- Implement multiple stock predictions at once.
- Deploy the model on cloud platforms.
- Improve UI with more interactive visualizations.



