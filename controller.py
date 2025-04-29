# controller.py
from model import Portfolio
from view import View

class PortfolioController:
    def __init__(self):
        self.portfolio = Portfolio()
        self.view = View()

    def run(self):
        while True:
            choice = self.view.main_menu()
            if choice == '1':
                asset = self.view.get_asset_input()
                self.portfolio.add_asset(**asset)
            elif choice == '2':
                data = self.portfolio.get_prices()
                self.view.plot_prices(data)
            elif choice == '3':
                table = self.portfolio.get_portfolio_table()
                self.view.display_table(table)
            elif choice == '4':
                asset_weights, sector_weights, class_weights = self.portfolio.calculate_weights()
                print("\n=== Asset Weights ===")
                self.view.display_summary(asset_weights)
                print("\n=== Sector Weights ===")
                self.view.display_summary(sector_weights)
                print("\n=== Asset Class Weights ===")
                self.view.display_summary(class_weights)
            elif choice == '5':
                simulations, metrics, dates_sim = self.portfolio.run_simulation()
                self.view.plot_simulation(simulations, dates_sim)
            elif choice == '6':
                simulations, metrics, dates_sim = self.portfolio.run_simulation()
                self.view.display_metrics(metrics)
            elif choice == 'q':
                break




