# Stock Market Predictor

This project utilizes machine learning to predict stock prices based on historical data. The model processes stock price data using features like closing prices, technical indicators (Simple Moving Averages, Volatility), and employs machine learning algorithms to predict future stock prices.

![Stock Market Bot]() ![Screenshot 2025-03-02 193133](https://github.com/user-attachments/assets/242a0582-973f-4c5b-8c85-02a3af1779d8)
 <!-- Make sure to add the image in the specified path -->

## Features:
- Fetches historical stock data for a specific company (default: Apple) using the `yfinance` library.
- Calculates technical indicators like **Simple Moving Averages (SMA)** and **Volatility**.
- Trains a **Ridge Regression** model to predict future stock prices.
- Provides performance evaluation with **Mean Absolute Error (MAE)** and **R² Score**.
- Displays a plot comparing actual vs predicted stock prices.

## Installation:

To get the project running locally on your machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sepideMarzi/stock-market-predictor.git
