# DSA210_TermProject

# Digital Panic and Market Dynamics: Turkey's Economic Transformation (2023–2025)

**DSA 210 - Introduction to Data Science**  
**Fall 2025–2026**  
**Sabancı University**  
**Student/ID: Simla Tükenmez / 32613**

---

## Project Overview

This project explores how major societal events and economic policy shifts in Turkey (2023–2025) influenced financial markets through the lens of both **traditional economic indicators** and **digital behavioral data**.  
It aims to understand whether **digital panic indicators** -derived from search trends, social media sentiment, and news tone— can explain stock market movements more effectively than traditional macroeconomic metrics.

**Central Question:**  
> Can digital panic indicators explain BIST100 movements better than traditional macroeconomic indicators alone?

---

## Motivation

The period between 2023–2025 marks one of the most turbulent and transformative eras in Turkey’s recent economic history, shaped by:

- **February 2023:** Kahramanmaraş earthquakes  
- **May 2023:** Presidential and parliamentary elections  
- **June 2023:** Economic management change and start of monetary tightening  
- **2023–2024:** Rapid interest rate hikes (8.5% → 50%)  
- **2025:** Start of rate cuts, signaling normalization  

These turning points allow the project to investigate:
1. How **non-economic shocks** (e.g., earthquakes, elections) affect market volatility  
2. Behavioral shifts **before and after policy regime changes**  
3. The value of **digital behavioral data** in explaining market sentiment  

---

## Research Questions

### Main Question
**Do digital panic indicators outperform traditional macroeconomic indicators in explaining BIST100 movements?**

### Supporting Questions
1. Did the 2023 earthquake or election uncertainty trigger short-term market panic?  
2. Can search volume and social sentiment anticipate exchange rate volatility?  
3. Does social media sentiment precede or predict BIST100 movements?  
4. How do market behaviors differ between tightening (2023–2024) and normalization (2025)?  

---

## Hypotheses

| ID | Hypothesis | Expected Direction |
|----|-------------|--------------------|
| H1 | Earthquake reduced BIST returns post-event | Negative |
| H2 | Election period increased volatility | Positive |
| H3 | Policy shift changed market behavior | Change expected |
| H4 | “Dolar kuru” search trends correlate with USD/TRY volatility | Relationship expected |
| H5 | Negative sentiment predicts market decline | Negative |
| H6 | Digital Panic Index improves explanatory power | Improvement expected |
| H7 | Market response differs across monetary regimes | Regime difference expected |

---

## Data Sources & Collection Plan

### 1. Traditional Financial Data

| Data | Source | Frequency | Status |
|------|---------|------------|--------|
| BIST100 Index | Yahoo Finance API | Daily | Planned |
| USD/TRY Exchange Rate | Yahoo Finance API | Daily | Planned |
| TCMB Policy Rate | TCMB EVDS | Monthly | To be collected |
| CPI (Inflation) | TÜİK | Monthly | To be collected |

**Note on EUR/TRY:**  
The EUR/TRY exchange rate was initially considered for inclusion. However, it was excluded at this stage due to its very high correlation (r > 0.95) with USD/TRY, which would create multicollinearity issues and add little additional explanatory power. Since USD/TRY already captures the dominant currency-driven market reactions, including both would unnecessarily increase model complexity.  
Nevertheless, EUR/TRY may later be incorporated for **robustness testing** — to verify whether the model’s results remain consistent when alternative exchange rate indicators are used.

---

### 2. Digital Behavioral Data

#### Google Trends
- **Planned Keywords (may change):** “borsa istanbul”, “dolar kuru”, “enflasyon”, “faiz”  
- **Frequency:** Weekly (2023–2025)  
- **Collection Method:** Manually downloaded from Google Trends platform  
- **Status:** Collected manually, to be cleaned and merged 
- **Purpose:** Gauge public economic anxiety and information-seeking behavior  

#### Social Media (Ekşi Sözlük)
- **Platform:** Ekşi Sözlük (Turkey’s largest discussion forum)  
- **Planned Topics (may change):** *borsa-istanbul*, *dolar*, *enflasyon*, *bist100*, *faiz*, *ekonomi*  
- **Method:** Custom Python scraper (`BeautifulSoup`)  
- **Purpose:** Perform sentiment analysis using Turkish BERT model  
- **Status:** To be expanded during data collection phase  

#### News Media Analysis (Planned)
- **Sources:** Turkish economic news outlets **such as Bloomberg HT, Anadolu Ajansı, Sözcü, Hürriyet**  
- **Target:** 500–1,000 headlines (2023–2025)  
- **Focus:** Key economic and political event periods (e.g., February 2023 earthquake, May 2023 elections, rate decisions)  
- **Method:** Semi-automated scraping and manual curation  
- **Goal:** Evaluate tone consistency and panic-inducing language across sources  

---

### 3. Event Variables

| Event | Date | Type |
|--------|------|------|
| Kahramanmaraş Earthquake | Feb 6, 2023 | Natural Disaster |
| Election Round 1 | May 14, 2023 | Political |
| Election Round 2 | May 28, 2023 | Political |
| Policy Team Change | Jun 3, 2023 | Policy |
| Rate Peak (50%) | Mar 21, 2024 | Monetary Policy |
| First Rate Cut | Jan 23, 2025 | Monetary Policy |

---

## Planned Methodology

These or similar methodologies are planned to be used depending on data quality and model performance.

### Phase 1: Data Collection
- Retrieve and clean datasets (financial, digital, macroeconomic)
- Handle missing values and outliers
- Merge into unified time series  

### Phase 2: Exploratory Data Analysis
- Time-series visualization with event markers  
- Correlation and volatility comparison  
- Sentiment distribution and panic keyword frequency  

### Phase 3: Modeling 
- **Model A:** Traditional indicators  
- **Model B:** Digital indicators  
- **Model C:** Combined model  
- Evaluate with R², RMSE, MAE, and feature importance  

---

## Technologies & Tools

`Python 3.9+`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`,  
`xgboost`, `statsmodels`, `transformers`, `BeautifulSoup`, `pytrends`

---

## Reproducibility

- All code will be developed in **Python 3.9+**  
- A `requirements.txt` file will include all dependencies  
- Jupyter notebooks will document the workflow step-by-step  
- Repository structure will ensure full reproducibility

---

## Data Sources

- [Yahoo Finance](https://finance.yahoo.com/)  
- [TCMB EVDS](https://evds2.tcmb.gov.tr/)  
- [TÜİK](https://www.tuik.gov.tr/)  
- [Google Trends](https://trends.google.com/)  
- [Ekşi Sözlük](https://eksisozluk.com/)  
- [News Outlets such as Bloomberg HT, Anadolu Ajansı, Sözcü, Hürriyet](https://www.bloomberght.com/) 

---

## Limitations & Future Work

- News data will initially be limited to selected Turkish outlets; expanding to other media could improve coverage.  
- Sentiment classification may face accuracy challenges in mixed-language content.  
- Incorporating **Twitter or Reddit** data could enhance the representativeness of digital panic indicators.  
- Additional macroeconomic variables (M2 supply, reserves) can be integrated in future versions.  

---

## Academic Integrity

This work is original and created for **DSA 210 – Introduction to Data Science**.  
AI tools (e.g., LLMs) are used only for writing assistance and debugging, following Sabancı University’s academic integrity policies.

---

## Author

**Simla Tükenmez**  
Sabancı University  
Fall 2025–2026  
DSA 210 – Introduction to Data Science  

---

**Last Updated:** October 30, 2025  
**Project Status:** Proposal Stage (Data collection planned)
