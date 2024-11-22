from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from stock_predictor import EnhancedStockPredictor
import yfinance as yf
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# File storage
PORTFOLIO_FILE = 'portfolio.json'
ALERTS_FILE = 'alerts.json'

# Create files if they don't exist
if not os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump([], f)

if not os.path.exists(ALERTS_FILE):
    with open(ALERTS_FILE, 'w') as f:
        json.dump([], f)

def load_portfolio():
    with open(PORTFOLIO_FILE, 'r') as f:
        return json.load(f)

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f)

def load_alerts():
    with open(ALERTS_FILE, 'r') as f:
        return json.load(f)

def save_alerts(alerts):
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts, f)

@app.route('/')
def serve_html():
    return send_from_directory('static', 'index.html')

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    try:
        portfolio = load_portfolio()
        return jsonify(portfolio)
    except Exception as e:
        print(f"Error getting portfolio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['POST'])
def add_to_portfolio():
    try:
        data = request.json
        if not data or not all(key in data for key in ['ticker', 'shares', 'purchasePrice']):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        portfolio = load_portfolio()
        new_stock = {
            'id': str(len(portfolio)),
            'ticker': data['ticker'].upper(),
            'shares': float(data['shares']),
            'purchasePrice': float(data['purchasePrice']),
            'added_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Verify the stock exists
        stock = yf.Ticker(new_stock['ticker'])
        history = stock.history(period='1d')
        if len(history) == 0:
            return jsonify({'success': False, 'error': 'Invalid stock ticker'}), 400
        
        portfolio.append(new_stock)
        save_portfolio(portfolio)
        return jsonify({'success': True, 'stock': new_stock})
    except Exception as e:
        print(f"Error adding to portfolio: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/portfolio/<stock_id>', methods=['PUT'])
def update_stock(stock_id):
    try:
        data = request.json
        if not data or not all(key in data for key in ['shares', 'purchasePrice']):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        portfolio = load_portfolio()
        updated = False
        for stock in portfolio:
            if stock['id'] == stock_id:
                stock['shares'] = float(data['shares'])
                stock['purchasePrice'] = float(data['purchasePrice'])
                updated = True
                break

        if not updated:
            return jsonify({'success': False, 'error': 'Stock not found'}), 404

        save_portfolio(portfolio)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating stock: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/portfolio/<stock_id>', methods=['DELETE'])
def delete_stock(stock_id):
    try:
        portfolio = load_portfolio()
        initial_length = len(portfolio)
        portfolio = [stock for stock in portfolio if stock['id'] != stock_id]
        
        if len(portfolio) == initial_length:
            return jsonify({'success': False, 'error': 'Stock not found'}), 404

        save_portfolio(portfolio)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting stock: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stock-price', methods=['POST'])
def get_stock_price():
    try:
        data = request.json
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Missing ticker symbol'}), 400

        ticker = data['ticker'].upper()
        stock = yf.Ticker(ticker)
        
        # Get current price
        history = stock.history(period='1d')
        if len(history) == 0:
            return jsonify({'error': f"No data available for {ticker}"}), 400
            
        current_price = history['Close'].iloc[-1]
        return jsonify({'price': current_price})
    except Exception as e:
        print(f"Error getting stock price: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    try:
        alerts = load_alerts()
        return jsonify(alerts)
    except Exception as e:
        print(f"Error getting alerts: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['POST'])
def add_alert():
    try:
        data = request.json
        if not data or not all(key in data for key in ['ticker', 'targetPrice', 'type']):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        # Verify the stock exists
        stock = yf.Ticker(data['ticker'])
        history = stock.history(period='1d')
        if len(history) == 0:
            return jsonify({'success': False, 'error': 'Invalid stock ticker'}), 400

        alerts = load_alerts()
        new_alert = {
            'id': str(len(alerts)),
            'ticker': data['ticker'].upper(),
            'targetPrice': float(data['targetPrice']),
            'type': data['type']
        }
        
        alerts.append(new_alert)
        save_alerts(alerts)
        return jsonify({'success': True, 'alert': new_alert})
    except Exception as e:
        print(f"Error adding alert: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/alerts/<alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    try:
        alerts = load_alerts()
        initial_length = len(alerts)
        alerts = [alert for alert in alerts if alert['id'] != alert_id]
        
        if len(alerts) == initial_length:
            return jsonify({'success': False, 'error': 'Alert not found'}), 404

        save_alerts(alerts)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting alert: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get('ticker').upper()
        days = int(data.get('days', 30))

        stock = yf.Ticker(ticker)
        history = stock.history(period="1mo")
        
        if history.empty:
            return jsonify({
                'success': False,
                'error': f"Invalid ticker symbol or no data available: {ticker}"
            })

        predictor = EnhancedStockPredictor(lookback_days=60, prediction_days=days)
        score = predictor.train(ticker)
        predictions = predictor.predict(ticker)

        info = stock.info
        company_name = info.get('longName', ticker) if info else ticker

        prediction_data = [
            {
                'date': index.strftime('%Y-%m-%d'),
                'price': float(row['Predicted_Close'])
            }
            for index, row in predictions.iterrows()
        ]

        return jsonify({
            'success': True,
            'predictions': prediction_data,
            'score': float(score),
            'company_name': company_name
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error processing request: {str(e)}"
        })
@app.route('/api/stock-details', methods=['POST'])
def get_stock_details():
    try:
        data = request.json
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Missing ticker symbol'}), 400

        ticker = data['ticker'].upper()
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1y")
        
        # Calculate basic metrics
        if len(history) > 0:
            volatility = history['Close'].pct_change().std() * (252 ** 0.5)  # Annualized volatility
            beta = None
            try:
                spy = yf.Ticker("SPY").history(period="1y")['Close'].pct_change()
                stock_returns = history['Close'].pct_change()
                beta = (stock_returns.cov(spy) / spy.var()).round(2)
            except:
                pass
        else:
            volatility = None
            beta = None

        return jsonify({
            'success': True,
            'details': {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'marketCap': info.get('marketCap', 'N/A'),
                'peRatio': info.get('forwardPE', 'N/A'),
                'dividendYield': info.get('dividendYield', 'N/A'),
                'beta': beta,
                'volatility': volatility,
                'price': info.get('currentPrice', 'N/A'),
                'priceHistory': history['Close'].tolist() if len(history) > 0 else [],
                'dates': [d.strftime('%Y-%m-%d') for d in history.index] if len(history) > 0 else []
            }
        })
    except Exception as e:
        print(f"Error getting stock details: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=8000)

    