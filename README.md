# 📈 Momentum Investing Dashboard

**E628: Data Science for Business — Final Group Project**

An interactive Dash dashboard for backtesting S&P 500 momentum strategies, built with Python, Plotly, and scikit-learn.

---

## Live App

Deployed on Render: **[your-app-name.onrender.com](https://your-app-name.onrender.com)**

---

## What It Does

- **4 momentum signal methods**: Classic 12-1, Risk-Adjusted Sharpe, Composite Multi-Horizon, Volatility-Filtered
- **Walk-forward backtesting** against S&P 500 benchmark (no look-ahead bias)
- **Parameter sensitivity**: live sweeps over lookback window and portfolio size
- **ML stock ranker**: Ridge, Random Forest, XGBoost trained with TimeSeriesSplit CV
- **6 interactive tabs**: Equity Curves | Drawdown & Risk | Signal Explorer | Parameter Sweep | ML Ranker | Portfolio

---

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/momentum-dashboard.git
cd momentum-dashboard
pip install -r requirements.txt
python app.py
```
Then open [http://localhost:8050](http://localhost:8050).

The cache file (`sp500_prices_cache.csv`) is included — no download needed on first run.

---

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**

The `render.yaml` is already configured:
- **Build**: `pip install -r requirements.txt`
- **Start**: `gunicorn app:server --workers 1 --threads 2 --timeout 120`
- **Plan**: Free tier

> ⚠️ **Free tier cold starts**: Render free tier spins down after 15 minutes of inactivity. First load after inactivity takes ~30 seconds as the app boots and runs backtests.

---

## Repo Structure

```
momentum-dashboard/
├── app.py                    # Dash application (main entry point)
├── sp500_prices_cache.csv    # Cached S&P 500 price data (~15 MB)
├── requirements.txt          # Python dependencies
├── render.yaml               # Render deployment config
├── .gitignore
└── README.md
```

---

## Data Sources

- **Price data**: Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance) (personal/academic use)
- **Constituents**: Wikipedia S&P 500 list (current constituents — survivorship bias applies)
- **Benchmark**: S&P 500 index (`^GSPC`) via yfinance

---

## Academic References

- Jegadeesh & Titman (1993) — *Returns to Buying Winners and Selling Losers*
- Daniel & Moskowitz (2016) — *Momentum Crashes*
- Asness, Moskowitz & Pedersen (2013) — *Value and Momentum Everywhere*
