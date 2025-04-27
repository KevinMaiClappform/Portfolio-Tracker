# view.py
import matplotlib.pyplot as plt
from tabulate import tabulate

class View:
    def main_menu(self):
        print("\n--- Portfolio Tracker CLI ---")
        print("1. Add asset")
        print("2. Show price graph")
        print("3. View portfolio table")
        print("4. Show weights")
        print("5. Run simulation")
        print("q. Quit")
        return input("Choose option: ")

    def get_asset_input(self):
        ticker = input("Ticker (e.g., PEP): ")
        sector = input("Sector: ")
        asset_class = input("Asset class: ")
        quantity = input("Quantity: ")
        purchase_price = input("Purchase price: ")
        return {
            'ticker': ticker,
            'sector': sector,
            'asset_class': asset_class,
            'quantity': quantity,
            'purchase_price': purchase_price
        }

    def plot_prices(self, data):
        if data.empty:
            print("No price data available.")
            return
        data.plot(title="Asset Prices")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True)
        plt.show()

    def display_table(self, df):
        if df.empty:
            print("No portfolio data available.")
            return
        print(tabulate(df, headers='keys', tablefmt='psql'))

    def display_summary(self, df):
        if df.empty:
            print("No weights data available.")
            return
        print(tabulate(df, headers='keys', tablefmt='grid'))

    def plot_simulation(self, simulations):
        if simulations.shape[0] == 0:
            print("No simulation data available.")
            return
        plt.figure(figsize=(12, 6))
        for i in range(100):
            plt.plot(simulations[:, i], alpha=0.1)
        plt.title("LightGBM Monte Carlo Simulation - Portfolio Value")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        plt.show()

    def display_metrics(self, metrics):
        if not metrics:
            print("No metrics to display.")
            return
        print("\nSimulation Insights:")
        for key, value in metrics.items():
            if 'VaR' in key:
                print(f"{key}: ${value:,.2f}")
            else:
                print(f"{key}: {value:.4f}")
