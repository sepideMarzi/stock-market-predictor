import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Fetch stock data (Example: Apple - AAPL)
def fetch_stock_data(ticker, period="5y"):
    print(f"Fetching {period} of stock data for {ticker}...")
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Prepare dataset
def prepare_data(data):
    print("Preparing data by adding technical indicators...")
    # Add some technical indicators (e.g., moving averages)
    data['SMA_10'] = data['Close'].rolling(window=10).mean()  # 10-day Simple Moving Average
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
    data['Volatility'] = data['Close'].rolling(window=10).std()  # 10-day volatility (standard deviation)
    data = data.dropna()  # Remove rows with NaN values resulting from the rolling window
    
    # Predict next 30 days
    data['Prediction'] = data['Close'].shift(-30)  # We are predicting the next 30 days' closing prices
    X = data[['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_50', 'Volatility']].values
    X = X[:-30]  # Remove last 30 NaN rows, which are caused by the prediction shift
    y = data['Prediction'].dropna().values  # Remove NaN values in target variable
    print(f"Prepared data with {len(X)} samples.")
    return X, y

# Train model
def train_model(X, y):
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling to normalize values between 0 and 1
    print("Scaling features using MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training the Ridge regression model...")
    model = Ridge(alpha=1.0)  # Using Ridge regression for better generalization
    model.fit(X_train_scaled, y_train)
    
    return model, X_test_scaled, y_test, scaler

# Predict and evaluate
def predict_and_evaluate(model, X_test, y_test):
    print("Making predictions and evaluating the model...")
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Print out the performance metrics
    print(f"Mean Absolute Error (MAE): {error:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='red')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.show()

# Run the program
def main():
    print("Welcome to the stock price prediction model!")
    
    ticker = "AAPL"  # You can change this to any stock symbol
    print(f"Let's start predicting stock prices for {ticker}.")
    
    # Fetch stock data
    data = fetch_stock_data(ticker)
    
    # Prepare the data
    X, y = prepare_data(data)
    
    # Train the model
    model, X_test, y_test, scaler = train_model(X, y)
    
    # Make predictions and evaluate
    predict_and_evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()
