# Portfolio Tracker CLI

Een command-line applicatie voor het beheren en analyseren van een beleggingsportfolio

---

## Functionaliteiten

- Assets toevoegen (ticker, sector, asset class, hoeveelheid, aankoopprijs)
- Koersgrafieken tonen (huidig & historisch via [yfinance](https://pypi.org/project/yfinance/))
- Portfolio-overzicht met actuele waarden
- Weging per asset, sector en asset class
- Monte Carlo simulatie (15 jaar, 100.000 paden) met **LightGBM machine learning**
- Simulatiestatistieken: Sharpe Ratio, Volatiliteit, Value-at-Risk (VaR)

---

## Installatie

1. Maak een virtuele omgeving
python -m venv venv

2. Activeer de virtuele omgeving

Voor macOS/Linux:
source venv/bin/activate

Voor Windows:
venv\Scripts\activate

3. Installeer dependencies
pip install -r requirements.txt

4. Start het programma via de terminal
python -i main.py

---


## CLI MENU

--- Portfolio Tracker CLI ---
1. Add asset

voorbeeld:


Ticker (e.g., PEP): AAPL
Sector: Information Technology
Asset class: Equity
Quantity: 15
Purchase price: 175

Ticker (e.g., PEP): PEP
Sector: Consumer Staples
Asset class: Equity
Quantity: 10
Purchase price: 170


2. Show price graph
3. View portfolio table
4. Show weights
5. Run simulation
q. Quit


---

## Opmerkingen

Een actieve internetverbinding is nodig voor het ophalen van prijsdata via yfinance.

De simulatie gebruikt machine learning (lightGBM) voor voorspelling van rendement en volatiliteit op basis van historische koersdata. Dit duurt ongeveer 1 minuut.

Pas wanneer je de simulatie plot weer sluit kan je in de terminal typen.


| Bestand                             | Functie                                        |
|-------------------------------------|------------------------------------------------|
| `main.py`                           | Startpunt van de applicatie                    |
| `controller.py`                     | Stuurt de gebruikersinteractie aan (MVC)       |
| `model.py`                          | Bevat logica voor portfolio en simulatie       |
| `view.py`                           | CLI interface en visualisaties                 |
| `requirements.txt`                  | Vereiste Python packages                       |
| `MachineLearning Return.ipynb`      | Onderzoek Machinelearning returns              |
| `MachineLearning Volatiliteit.ipynb`| Onderzoek Machinelearning volatiliteit         |
| `context.md`                        | Context van mijn onderzoek                     |
