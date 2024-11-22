from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class EnhancedStockPredictor:
    def __init__(self, lookback_days=60, prediction_days=30):
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler()
        self.model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
    def fetch_data(self, ticker):
        try:
            # Get 5 years of data instead of 3 to ensure enough history
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*5)
            
            # Fetch stock data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            # Ensure we have enough data
            if len(df) < (self.lookback_days + self.prediction_days + 50):  # Added buffer
                raise ValueError(f"Not enough historical data for {ticker}")
            
            # Calculate basic features
            df['Returns'] = df['Close'].pct_change()
            
            # Moving averages
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            
            # Volatility
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Volume features
            df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            
            # Drop any NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")
    
    def prepare_data(self, df):
        try:
            # Select features for prediction
            features = ['Close', 'Volume', 'Returns', 'MA_5', 'MA_20', 
                       'Volatility', 'Volume_MA_5', 'Volume_MA_20']
            
            # Ensure all required features exist
            for feature in features:
                if feature not in df.columns:
                    raise ValueError(f"Missing required feature: {feature}")
            
            # Scale the features
            scaled_data = self.scaler.fit_transform(df[features])
            
            X, y = [], []
            for i in range(self.lookback_days, len(df) - self.prediction_days):
                X.append(scaled_data[i - self.lookback_days:i])
                y.append(df['Close'].iloc[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            raise Exception(f"Error preparing data: {str(e)}")
    
    def train(self, ticker):
        try:
            # Fetch and prepare data
            df = self.fetch_data(ticker)
            X, y = self.prepare_data(df)
            
            # Reshape data for models
            X_reshaped = X.reshape(X.shape[0], -1)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_reshaped, y, test_size=0.2, random_state=42
            )
            
            # Train models
            self.model_rf.fit(X_train, y_train)
            self.model_gb.fit(X_train, y_train)
            
            # Calculate combined score
            rf_score = self.model_rf.score(X_test, y_test)
            gb_score = self.model_gb.score(X_test, y_test)
            return (rf_score + gb_score) / 2
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def predict(self, ticker):
        try:
            # Fetch recent data
            df = self.fetch_data(ticker)
            
            # Get the most recent data for prediction
            recent_data = df.iloc[-self.lookback_days:]
            
            # Prepare features for prediction
            features = ['Close', 'Volume', 'Returns', 'MA_5', 'MA_20', 
                       'Volatility', 'Volume_MA_5', 'Volume_MA_20']
            recent_scaled = self.scaler.transform(recent_data[features])
            recent_reshaped = recent_scaled.reshape(1, -1)
            
            # Make predictions with both models
            rf_pred = self.model_rf.predict([recent_reshaped[0]])
            gb_pred = self.model_gb.predict([recent_reshaped[0]])
            
            # Average the predictions
            final_pred = (rf_pred + gb_pred) / 2
            
            # Create future dates
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                       periods=self.prediction_days, 
                                       freq='B')
            
            # Create prediction DataFrame with a trend
            predictions = []
            current_price = df['Close'].iloc[-1]
            price_change = (final_pred[0] - current_price) / self.prediction_days
            
            for i in range(self.prediction_days):
                next_price = current_price + (price_change * (i + 1))
                predictions.append(next_price)
            
            return pd.DataFrame(
                predictions,
                index=future_dates,
                columns=['Predicted_Close']
            )
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")

def main():
    predictor = EnhancedStockPredictor(lookback_days=60, prediction_days=30)
    ticker = "AAPL"
    score = predictor.train(ticker)
    predictions = predictor.predict(ticker)
    print(f"Model Score: {score}")
    print("\nPredictions:")
    print(predictions)

if __name__ == "__main__":
    main()