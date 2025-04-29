Dit is mijn context.md file waar ik uitleg hoe ik te werk ging en wat ik allemaal onderzocht heb.

Voor deze opdracht heb ik meerdere machine learning modellen getest om toekomstige rendementen en volatiliteit te voorspellen.  
Ik heb gewerkt met drie verschillende typen modellen:

- **Lineaire regressie**: Elastic Net
- **Decision Tree**: Random Forest
- **Gradient Boosting**: XGBoost en LightGBM

Uit mijn experimenten blijkt:

- **Elastic Net** presteerde het slechtst, zoals verwacht, omdat aandelenkoersen vaak een niet-lineaire structuur vertonen.
- **Random Forest** presteerde beter dan Elastic Net, maar vertoonde regelmatig overfitting en had een wat hogere RMSE.
- **XGBoost en LightGBM** leverden de beste prestaties: een lage RMSE en een hoge R²-score, met minder overfitting dan Random Forest.

Ik heb ook een **Likelihood Ratio Test** uitgevoerd om de modellen onderling te vergelijken.

---

## Voordelen van Machine Learning

- Geen aannames nodig over de verdeling van data.
- Geen noodzaak tot complexe multivariate modellen (bijvoorbeeld meerdere aandelen tegelijk modelleren).
- Geen noodzaak voor autocorrelatie-analyses.

---

## Onderzochte Aandelen

Voor elk machine learning model heb ik de prestaties getest op 10 verschillende aandelen:

- 5 aandelen die negatief zijn beïnvloed door de **tariffs van Trump**.
- 5 aandelen die positief zijn beïnvloed door de **tariffs van Trump**.

---

## Verbeterpunten

### Grid Search en Cross-Validation

Om de modelprestaties te verbeteren, heb ik geëxperimenteerd met:

- Het uitbreiden van de hyperparametergrid.
- Het verhogen van het aantal folds in cross-validation.

Let op: bij een te complex model kan weer **overfitting** ontstaan.

### Overfitting Tegengaan

Om overfitting te minimaliseren, heb ik de volgende strategieën onderzocht:

1. **Verhoogde Regularisatie**: 
   - Tunen van parameters zoals `reg_alpha`, `reg_lambda`, en `max_depth`.
2. **Verlaging van modelcomplexiteit**: 
   - Bijvoorbeeld door `max_depth`, `n_estimators` en `min_child_weight` te verlagen.
3. **Vroegtijdig stoppen**:
   - Om te voorkomen dat modellen te lang trainen en overfitten.

---


## Monte Carlo Simulatie

Voor de Monte Carlo simulatie zijn **twee aparte LightGBM modellen** gebruikt:

- Eén voor het voorspellen van toekomstige rendementen.
- Eén voor het voorspellen van toekomstige volatiliteit.

Dit stelt de simulatie in staat om optimaal gebruik te maken van moderne machine learning technieken, zoals in de opdracht werd gevraagd.







## Feature Importance / SHAP Analyse

### Volatiliteit

De SHAP-analyse van volatiliteit toonde aan:

- Het **21-daags voortschrijdend gemiddelde (MA_21)** had de grootste invloed op voorspellingen.
- De **RSI-indicator** volgde als tweede belangrijkste variabele, mogelijk door marktdruk bij overbought-condities.
- Indicatoren zoals historische volatiliteit en MACD bleken minder voorspellend.

### Rendement

Voor de voorspelling van toekomstige rendementen bleek:

- De **return over de laatste 5 dagen** het meest informatief (momentum-effect).
- De **1-daagse return** had ook voorspellende waarde, maar in mindere mate.
- Trendvolgende indicatoren zoals MA(5) en MA(10) hadden weinig unieke voorspellende waarde.

---


## Feature Engineering & Modelkeuze – Uitleg

### Waarom deze features?
De gekozen features zijn gebaseerd op bewezen signalen uit de financiële literatuur en technische analyse. Ze zijn bedoeld om zowel **trend**, **momentum** als **risico** te vangen:

| Feature                    | Reden voor keuze                                              |
|----------------------------|---------------------------------------------------------------|
| `return_1d`, `return_5d`   | Captureren kortetermijnmomentum (prijsverandering over 1 of 5 dagen). |
| `ma_5`, `ma_10`            | Trendindicatoren: detecteren voortschrijdend koersgemiddelde. |
| `vol_5d`, `vol_10d`, `vol_21d` | Schatting van volatiliteit: cruciaal voor risicoanalyse.    |
| `abs_return_1d`, `squared_return_1d` | Alternatieve schattingen van volatiliteit; helpen bij robuustheid. |
| `target_return`            | Gebaseerd op rolling average om noise in individuele returns te dempen. |

---

### Waarom  log returns?
- **Log returns zijn additief over tijd** → handig voor cumulatieve simulaties.
- Ze normaliseren extreme waarden beter dan gewone returns.
- Worden standaard gebruikt in kwantitatieve finance.

---

### Waarom smoothing voor targets?
Het gebruik van een **rolling mean voor target_return** helpt bij:
- Vermijden van overfitting op ruis.
- Beter generaliseerbare voorspellingen.
- Meer stabiliteit in de simulatie op lange termijn.

---

Deze keuzes zijn essentieel om een portfolio simulatie te bouwen die realistische lange termijn rendementen en risico’s inschat, zonder te vervallen in extreme overfitting of onrealistische volatiliteit.




