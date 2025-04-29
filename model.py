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
        data = yf.download(tickers, start='2015-01-01', end='2025-04-29', auto_adjust=True)['Close'].dropna()
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
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        df['weight'] = df['current_value'] / df['current_value'].sum()

        # Per asset (wat je nu al toont)
        asset_weights = df[['ticker', 'weight', 'sector', 'asset_class']]

        # Per sector
        sector_weights = df.groupby('sector')['current_value'].sum()
        sector_weights = (sector_weights / df['current_value'].sum()).reset_index()
        sector_weights.columns = ['sector', 'weight']

        # Per asset class
        class_weights = df.groupby('asset_class')['current_value'].sum()
        class_weights = (class_weights / df['current_value'].sum()).reset_index()
        class_weights.columns = ['asset_class', 'weight']

        return asset_weights, sector_weights, class_weights


    def run_simulation(self):
        df = self.get_prices()
        if df.empty:
            return np.zeros((0, 0))

        ticker = self.assets[0]['ticker']  # Simpelweg de eerste asset pakken
        price_data = df[ticker]



        # Feature Engineering
        features = pd.DataFrame(index=price_data.index)
        features['return_1d'] = price_data.pct_change(1)
        features['return_5d'] = price_data.pct_change(5)
        features['ma_5'] = price_data.rolling(window=5).mean()
        features['ma_10'] = price_data.rolling(window=10).mean()
        features['vol_5d'] = price_data.pct_change().rolling(window=5).std()
        features['abs_return_1d'] = features['return_1d'].abs()
        features['squared_return_1d'] = features['return_1d']**2
        features['vol_10d'] = price_data.pct_change().rolling(10).std()
        features['vol_21d'] = price_data.pct_change().rolling(21).std()
        features = features.dropna()

        # Targets
        y_return = features['return_1d'].shift(-1).dropna()
        y_volatility = features['vol_5d'].shift(-1).dropna()
        X = features.loc[y_return.index]

        # Train/test split
        X_train, X_test, y_return_train, y_return_test = train_test_split(X, y_return, test_size=0.2, shuffle=False)
        _, _, y_vol_train, y_vol_test = train_test_split(X, y_volatility, test_size=0.2, shuffle=False)

        # Train LightGBM modellen
        model_return = LGBMRegressor(verbose=-1)
        model_return.fit(X_train, y_return_train)

        model_vol = LGBMRegressor(verbose=-1)
        model_vol.fit(X_train, y_vol_train)

        # Voorspellingen
        y_return_pred = model_return.predict(X_test)
        y_vol_pred = model_vol.predict(X_test)

        # Simulatie parameters
        start_value = 10000
        n_years = 15
        n_days = n_years * 252
        n_paths = 100000

        simulations = np.zeros((n_days, n_paths))
        simulations[0] = start_value

        # Create real dates for simulation
        start_date_sim = pd.to_datetime('2025-01-01')
        dates_sim = pd.date_range(start=start_date_sim, periods=n_days, freq='B')

        predicted_returns = np.resize(y_return_pred, n_days)
        predicted_vols = np.resize(y_vol_pred, n_days)

        np.random.seed(42)
        for t in range(1, n_days):
            Z = np.random.normal(0, 1, n_paths)
            drift = predicted_returns[t] - 0.5 * predicted_vols[t] ** 2
            shock = predicted_vols[t] * Z
            simulations[t] = simulations[t-1] * np.exp(drift + shock)

        # Analyse
        final_values = simulations[-1]
        annualized_returns = (final_values / start_value) ** (1 / n_years) - 1
        mean_annual_return = np.mean(annualized_returns)

        log_returns = np.log(simulations[1:] / simulations[:-1])
        portfolio_volatility = np.std(log_returns, axis=0) * np.sqrt(252)
        mean_annual_volatility = np.mean(portfolio_volatility)

        sharpe_ratio = mean_annual_return / mean_annual_volatility

        losses = start_value - final_values
        var_5 = np.percentile(losses, 5)

        metrics = {
            'Mean Annual Return': mean_annual_return,
            'Mean Annual Volatility': mean_annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            '5% VaR': var_5
        }

        return simulations, metrics, dates_sim