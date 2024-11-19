from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from stock_predictor import EnhancedStockPredictor  # Changed this line
import yfinance as yf
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_html():
    return send_from_directory('static', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get('ticker').upper()
        days = int(data.get('days', 30))

        # Verify ticker exists by trying to get its history
        stock = yf.Ticker(ticker)
        history = stock.history(period="1mo")
        
        if history.empty:
            return jsonify({
                'success': False,
                'error': f"Invalid ticker symbol or no data available: {ticker}"
            })

        predictor = EnhancedStockPredictor(lookback_days=60, prediction_days=days)  # Changed this line
        score = predictor.train(ticker)
        predictions = predictor.predict(ticker)

        # Get company info
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
        print(f"Error processing request: {str(e)}")  # For debugging
        return jsonify({
            'success': False,
            'error': f"Error processing request: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True, port=8000)