# controller.py
from model import Portfolio
from view import View

class PortfolioController:
    def __init__(self):
        self.portfolio = Portfolio()
        self.view = View()
        self.preload_pep()

    def preload_pep(self):
        # Automatisch PepsiCo (PEP) toevoegen bij opstarten
        self.portfolio.add_asset(
            ticker='PEP',
            sector='Consumer Staples',
            asset_class='Equity',
            quantity=10,  # voorbeeld aantal aandelen
            purchase_price=170.00  # voorbeeld aankoopprijs
        )

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
                summary = self.portfolio.calculate_weights()
                self.view.display_summary(summary)
            elif choice == '5':
                sim = self.portfolio.run_simulation()
                self.view.plot_simulation(sim)
            elif choice == 'q':
                break
