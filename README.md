# Stock Market Analysis using Python and Machine Learning

Web deployment of the existing supervised machine-learning stock analysis project.

## Models
- Logistic Regression
- Gradient Boosting

## Data
Historical market data is retrieved with yfinance.

## Features
The project uses returns, moving averages, volatility, RSI, MACD, and volume-trend features.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000

## Render

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
gunicorn app:app
```
