# app.py
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import yfinance as yf
from lightgbm import LGBMRegressor
import plotly.graph_objs as go
import plotly
import json
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Portfolio opslag
portfolio = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate')
def simulate():
    return render_template('simulate.html')



# Homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    global portfolio
    if request.method == 'POST':
        ticker = request.form['ticker']
        quantity = float(request.form['quantity'])
        purchase_price = float(request.form['purchase_price'])
        sector = request.form['sector']
        asset_class = request.form['asset_class']

        portfolio.append({
            'ticker': ticker.upper(),
            'quantity': quantity,
            'purchase_price': purchase_price,
            'sector': sector,
            'asset_class': asset_class
        })
        return redirect(url_for('index'))

    return render_template('index.html', portfolio=portfolio)

# Simulatie pagina
@app.route('/simulate')
def simulate():
    if not portfolio:
        return redirect(url_for('index'))

    ticker = portfolio[0]['ticker']

    df = yf.download(ticker, start='2018-01-01', end='2024-01-01', auto_adjust=True)['Close'].dropna()

    features = pd.DataFrame(index=df.index)
    features['return_1d'] = df.pct_change(1)
    features['return_5d'] = df.pct_change(5)
    features['ma_5'] = df.rolling(window=5).mean()
    features['ma_10'] = df.rolling(window=10).mean()
    features['vol_5d'] = df.pct_change().rolling(window=5).std()
    features = features.dropna()

    y_return = features['return_1d'].shift(-1).dropna()
    y_volatility = features['vol_5d'].shift(-1).dropna()
    X = features.loc[y_return.index]

    X_train, X_test, y_return_train, y_return_test = train_test_split(X, y_return, test_size=0.2, shuffle=False)
    _, _, y_vol_train, y_vol_test = train_test_split(X, y_volatility, test_size=0.2, shuffle=False)

    model_return = LGBMRegressor()
    model_return.fit(X_train, y_return_train)

    model_vol = LGBMRegressor()
    model_vol.fit(X_train, y_vol_train)

    y_return_pred = model_return.predict(X_test)
    y_vol_pred = model_vol.predict(X_test)

    start_value = 10000
    n_years = 15
    n_days = n_years * 252
    n_paths = 300

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

    fig = go.Figure()
    for i in range(100):
        fig.add_trace(go.Scatter(x=list(range(n_days)), y=simulations[:, i], mode='lines', line=dict(width=1), opacity=0.2))
    fig.update_layout(title=f"LightGBM Monte Carlo Simulation - {ticker}", xaxis_title="Days", yaxis_title="Portfolio Value ($)")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('simulate.html', graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
