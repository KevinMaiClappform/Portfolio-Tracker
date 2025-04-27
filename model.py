import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

class Portfolio:
    def __init__(self):
        self.assets = []

    def add_asset(self, ticker, sector, asset_class, quantity, purchase_price):
        self.assets.append({
            'ticker': ticker,
            'sector': sector,
            'asset_class': asset_class,
            'quantity': float(quantity),
            'purchase_price': float(purchase_price)
        })

    def get_prices(self):
        tickers = [asset['ticker'] for asset in self.assets]
        if not tickers:
            return pd.DataFrame()
        data = yf.download(tickers, period="1y", auto_adjust=True)['Close']
        return data

    def get_portfolio_table(self):
        prices = self.get_prices()
        table = []
        for asset in self.assets:
            ticker = asset['ticker']
            if ticker in prices.columns:
                current_price = prices[ticker].iloc[-1]
                transaction_value = asset['quantity'] * asset['purchase_price']
                current_value = asset['quantity'] * current_price
                table.append({
                    'ticker': ticker,
                    'sector': asset['sector'],
                    'asset_class': asset['asset_class'],
                    'quantity': asset['quantity'],
                    'purchase_price': asset['purchase_price'],
                    'current_price': current_price,
                    'transaction_value': transaction_value,
                    'current_value': current_value
                })
        return pd.DataFrame(table)

    def calculate_weights(self):
        df = self.get_portfolio_table()
        if df.empty:
            return pd.DataFrame()
        df['weight'] = df['current_value'] / df['current_value'].sum()
        return df[['ticker', 'weight', 'sector', 'asset_class']]

    def run_simulation(self):
        df = self.get_prices()
        if df.empty:
            return np.zeros((0, 0))

        ticker = self.assets[0]['ticker']  # Simpelweg de eerste asset pakken
        price_data = df[ticker]

        features = pd.DataFrame(index=price_data.index)
        features['return_1d'] = price_data.pct_change(1)
        features['return_5d'] = price_data.pct_change(5)
        features['ma_5'] = price_data.rolling(window=5).mean()
        features['ma_10'] = price_data.rolling(window=10).mean()
        features['vol_5d'] = price_data.pct_change().rolling(window=5).std()
        features = features.dropna()

        # Targets
        y_return = features['return_1d'].shift(-1).dropna()
        y_volatility = features['vol_5d'].shift(-1).dropna()
        X = features.loc[y_return.index]

        # Train/test split
        X_train, X_test, y_return_train, y_return_test = train_test_split(X, y_return, test_size=0.2, shuffle=False)
        _, _, y_vol_train, y_vol_test = train_test_split(X, y_volatility, test_size=0.2, shuffle=False)

        # Train LightGBM modellen
        model_return = LGBMRegressor()
        model_return.fit(X_train, y_return_train)

        model_vol = LGBMRegressor()
        model_vol.fit(X_train, y_vol_train)

        # Voorspellingen
        y_return_pred = model_return.predict(X_test)
        y_vol_pred = model_vol.predict(X_test)

        # Simulatie parameters
        start_value = 10000
        n_years = 15
        n_days = n_years * 252
        n_paths = 1000

        simulations = np.zeros((n_days, n_paths))
        simulations[0] = start_value

        predicted_returns = np.resize(y_return_pred, n_days)
        predicted_vols = np.resize(y_vol_pred, n_days)

        np.random.seed(42)
        for t in range(1, n_days):
            Z = np.random.normal(0, 1, n_paths)
            drift = predicted_returns[t] - 0.5 * predicted_vols[t] ** 2
            shock = predicted_vols[t] * Z
            simulations[t] = simulations[t-1] * np.exp(drift + shock)

        return simulations