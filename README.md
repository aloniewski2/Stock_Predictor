# Stock Portfolio & Predictor

A web application for managing a stock portfolio and predicting future stock prices. 

## Features
- Add, edit, and delete stocks in your portfolio
- View current portfolio value and total gain/loss
- Set price alerts for stocks
- Predict future stock prices using machine learning

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt` 
3. Run the application: `python app.py`
4. Access the app at `http://localhost:8000`

## Usage
- Add stocks to your portfolio by entering the ticker symbol, number of shares, and purchase price
- View and manage your portfolio in the "Portfolio" section
- Set price alerts in the "Price Alerts" section to be notified when a stock reaches a certain price
- Use the "Stock Predictor" to forecast future prices for a given stock

## Files
- `app.py` - Main Flask application 
- `stock_predictor.py` - Machine learning model for stock price prediction
- `index.html` - Front-end HTML file
- `portfolio.json` - Stores portfolio data
- `alerts.json` - Stores price alert data

## Technologies
- Python
- Flask
- yfinance
- scikit-learn
- HTML/CSS
