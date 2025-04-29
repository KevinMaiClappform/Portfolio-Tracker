# view.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tabulate import tabulate

class View:
    def main_menu(self):
        print("\n--- Portfolio Tracker CLI ---")
        print("1. Add asset")
        print("2. Show price graph")
        print("3. View portfolio table")
        print("4. Show weights")
        print("5. Run simulation")
        print("6. Show metrics")
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

        total_value = df['current_value'].sum()
        print(f"\nTotale portefeuillewaarde: €{total_value:,.2f}")


    def display_summary(self, df):
        if df.empty:
            print("No weights data available.")
            return
        print(tabulate(df, headers='keys', tablefmt='grid'))

    def plot_simulation(self, simulations, dates_sim):

        n_years = 15
        n_days = n_years * 252

        # Create real dates for simulation
        start_date_sim = pd.to_datetime('2025-04-29')
        dates_sim = pd.date_range(start=start_date_sim, periods=n_days, freq='B')

        if simulations.shape[0] == 0:
            print("No simulation data available.")
            return

        percentiles = [5, 25, 50, 75, 95]
        fan_chart = np.percentile(simulations, percentiles, axis=1)

        plt.figure(figsize=(14, 7))

        # Plot Monte Carlo simulation paths
        for i in range(100):
            plt.plot(dates_sim, simulations[:, i], alpha=0.05, color='gray')

        # Overlay percentile ranges
        plt.plot(dates_sim, fan_chart[2], label='Median', color='blue', linewidth=2)
        plt.fill_between(dates_sim, fan_chart[0], fan_chart[-1], color='lightblue', alpha=0.3, label='5%-95% Range')
        plt.fill_between(dates_sim, fan_chart[1], fan_chart[-2], color='blue', alpha=0.5, label='25%-75% Range')

        # Titles and labels
        plt.title("Monte Carlo Simulation for Portfolio\n(LightGBM Returns + LightGBM Volatility)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)

        # Format x-axis
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


    def display_metrics(self, metrics):
        if not metrics:
            print("No metrics to display.")
            return

        # Prepare data for table display
        table_data = []
        for key, value in metrics.items():
            if 'VaR' in key:
                formatted_value = f"${value:,.2f}"
            else:
                formatted_value = f"{value:.4f}"
            table_data.append([key, formatted_value])

        print("\nSimulation Insights (Performance Metrics):")
        print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))

