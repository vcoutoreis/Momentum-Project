"""
Momentum Investing Dashboard — Dash App
E628: Data Science for Business
All calculations in Python/pandas; UI in Dash + Plotly.
"""

import warnings, time, os
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import joblib
from bs4 import BeautifulSoup

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc

# ── Constants ──────────────────────────────────────────────────────────────────
SEED          = 42
YEARS         = 5
NUM_STOCKS    = 20
LOOKBACK_LONG = 252
LOOKBACK_SKIP = 21
VOL_WINDOW    = 63
CACHE_FILE    = Path(__file__).parent / "sp500_prices_cache.csv"

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=YEARS * 365 + 90)).strftime("%Y-%m-%d")

# Finance palette
GREEN, RED, BLUE, PURPLE, AMBER, GREY = (
    "#2ECC71", "#E74C3C", "#3498DB", "#9B59B6", "#F39C12", "#95A5A6"
)
METHOD_LABELS = {
    "classic":             "Classic (12-1)",
    "risk_adjusted":       "Risk-Adjusted",
    "composite":           "Composite",
    "volatility_filtered": "Vol-Filtered",
}
METHOD_COLORS = {
    "classic":             GREEN,
    "risk_adjusted":       PURPLE,
    "composite":           AMBER,
    "volatility_filtered": RED,
    "S&P 500":             BLUE,
}

# ── Data Loading ───────────────────────────────────────────────────────────────
def download_sp500_constituents():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "MomentumStrategy/1.0"}
    response = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", {"id": "constituents"}) or soup.find("table")
    df = pd.read_html(str(table))[0]
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "symbol" in cl or "ticker" in cl:         col_map[col] = "ticker"
        elif "security" in cl or "company" in cl:    col_map[col] = "company"
        elif "sub" in cl:                             col_map[col] = "gics_subsector"
        elif "sector" in cl or "industry" in cl:     col_map[col] = "gics_sector"
    df = df.rename(columns=col_map)
    cols = [c for c in ["ticker","company","gics_sector","gics_subsector"] if c in df.columns]
    df = df[cols].assign(ticker=lambda x: x["ticker"].str.replace(".", "-", regex=False))
    return df


def download_prices(tickers, start, end, batch_size=100):
    frames = []
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    for batch in batches:
        try:
            raw   = yf.download(batch, start=start, end=end,
                                 auto_adjust=True, progress=False, threads=True)
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            frames.append(close)
        except Exception:
            pass
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices.columns = [str(c) for c in prices.columns]
    return prices


def load_data():
    constituents = download_sp500_constituents()
    if CACHE_FILE.exists():
        prices_raw = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    else:
        all_tickers = constituents["ticker"].tolist() + ["^GSPC"]
        prices_raw  = download_prices(all_tickers, START_DATE, END_DATE)
        prices_raw.to_csv(CACHE_FILE)

    bench_series = prices_raw["^GSPC"].copy() if "^GSPC" in prices_raw.columns else None
    stocks = prices_raw.drop(columns=["^GSPC"], errors="ignore")
    stocks = stocks.loc[:, stocks.notna().mean() >= 0.70].ffill(limit=5)

    bench_prices  = bench_series.dropna()
    bench_returns = np.log(bench_prices / bench_prices.shift(1)).to_frame("SP500")

    sector_map = (
        constituents.set_index("ticker")[["gics_sector","company"]]
        .reindex(stocks.columns)
        .fillna("Unknown")
    )
    return stocks, bench_prices, bench_returns, sector_map


# ── MomentumStrategy class ─────────────────────────────────────────────────────
class MomentumStrategy:
    def __init__(self, lookback_long=LOOKBACK_LONG, lookback_skip=LOOKBACK_SKIP,
                 vol_window=VOL_WINDOW, num_stocks=NUM_STOCKS):
        self.lookback_long = lookback_long
        self.lookback_skip = lookback_skip
        self.vol_window    = vol_window
        self.num_stocks    = num_stocks

    def _window(self, prices, date):
        idx = prices.index.get_indexer([date], method="ffill")[0]
        return prices.iloc[max(0, idx - self.lookback_long): idx + 1]

    def calculate_momentum_score(self, prices, date):
        w = self._window(prices, date)
        if len(w) < self.lookback_long * 0.8:
            return pd.Series(dtype=float)
        price_end   = w.iloc[-(self.lookback_skip + 1)]
        price_start = w.iloc[0]
        return (price_end / price_start - 1).dropna()

    def calculate_risk_adjusted_momentum(self, prices, date):
        w       = self._window(prices, date)
        log_ret = np.log(w / w.shift(1)).dropna()
        if len(log_ret) < 60:
            return pd.Series(dtype=float)
        ann_ret = log_ret.mean() * 252
        ann_vol = log_ret.std()  * np.sqrt(252)
        return (ann_ret / ann_vol.replace(0, np.nan)).dropna()

    def calculate_composite_momentum(self, prices, date):
        horizons = [21, 63, 126, 252]
        weights  = [0.20, 0.30, 0.30, 0.20]
        idx      = prices.index.get_indexer([date], method="ffill")[0]
        signals  = []
        for h in horizons:
            if idx < h * 0.8:
                continue
            ret = prices.iloc[idx] / prices.iloc[idx - h] - 1
            z   = (ret - ret.mean()) / ret.std()
            signals.append(z)
        if not signals:
            return pd.Series(dtype=float)
        w_use = np.array(weights[:len(signals)])
        w_use /= w_use.sum()
        composite = pd.concat(signals, axis=1).mul(w_use, axis=1).sum(axis=1)
        return composite.dropna()

    def calculate_volatility_filtered_momentum(self, prices, date):
        mom = self.calculate_momentum_score(prices, date)
        idx = prices.index.get_indexer([date], method="ffill")[0]
        w   = prices.iloc[max(0, idx - self.vol_window): idx]
        vol = np.log(w / w.shift(1)).std() * np.sqrt(252)
        return mom.reindex(vol[vol <= vol.quantile(0.80)].index).dropna()

    def select_stocks(self, prices, date, method="composite"):
        dispatch = {
            "classic":             self.calculate_momentum_score,
            "risk_adjusted":       self.calculate_risk_adjusted_momentum,
            "composite":           self.calculate_composite_momentum,
            "volatility_filtered": self.calculate_volatility_filtered_momentum,
        }
        scores = dispatch[method](prices, date)
        return scores.nlargest(self.num_stocks).index.tolist()

    def backtest(self, prices, bench_daily, method="composite"):
        rebal_dates = prices.resample("ME").last().index
        min_start   = prices.index[self.lookback_long]
        rebal_dates = rebal_dates[rebal_dates >= min_start]
        # snap to trading days
        snap_idx    = prices.index.get_indexer(rebal_dates, method="ffill")
        rebal_dates = prices.index[snap_idx[snap_idx >= 0]]

        monthly_rets, holdings_log = [], []
        for i in range(len(rebal_dates) - 1):
            signal_date = rebal_dates[i]
            hold_end    = rebal_dates[i + 1]
            tickers     = self.select_stocks(prices, signal_date, method=method)
            if not tickers:
                continue
            hold    = prices.loc[signal_date:hold_end, tickers]
            ret_m   = hold.pct_change().mean(axis=1)
            total_r = (1 + ret_m).prod() - 1
            monthly_rets.append({"date": hold_end, "return": total_r})
            holdings_log.append({"date": signal_date, "tickers": tickers})

        ret_s  = pd.DataFrame(monthly_rets).set_index("date")["return"]
        equity = (1 + ret_s).cumprod()
        stats_ = self._calc_stats(ret_s, equity, bench_daily)
        return equity, stats_, pd.DataFrame(holdings_log)

    @staticmethod
    def _calc_stats(returns, equity, bench_daily):
        ann_ret = (1 + returns.mean()) ** 12 - 1
        ann_vol = returns.std() * np.sqrt(12)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        roll_max = equity.cummax()
        dd       = equity / roll_max - 1
        max_dd   = dd.min()
        calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
        win_rate = (returns > 0).mean()
        bench_m  = bench_daily.resample("ME").apply(lambda x: (1 + x).prod() - 1).squeeze()
        al = pd.concat([returns, bench_m], axis=1, join="inner")
        al.columns = ["port", "bench"]
        if len(al) > 5:
            cov   = al.cov()
            beta  = cov.iloc[0, 1] / al["bench"].var()
            b_ann = (1 + al["bench"].mean()) ** 12 - 1
            alpha = ann_ret - beta * b_ann
        else:
            beta = alpha = np.nan
        return {
            "Annual Return (%)":  round(ann_ret  * 100, 2),
            "Annual Vol (%)":     round(ann_vol  * 100, 2),
            "Sharpe Ratio":       round(sharpe,         3),
            "Max Drawdown (%)":   round(max_dd   * 100, 2),
            "Calmar Ratio":       round(calmar,         3),
            "Win Rate (%)":       round(win_rate * 100, 1),
            "Beta":               round(beta,           3),
            "Alpha (%)":          round(alpha    * 100, 2),
            "Total Return (%)":   round((equity.iloc[-1] - 1) * 100, 2),
        }


# ── ML helpers ────────────────────────────────────────────────────────────────
FEATURE_COLS = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_3m", "skew_3m"]
FEAT_LABELS  = {
    "ret_1m":  "1m momentum", "ret_3m":  "3m momentum",
    "ret_6m":  "6m momentum", "ret_12m": "12m momentum",
    "vol_3m":  "3m volatility", "skew_3m": "3m return skew",
}

def build_ml_dataset(prices, returns, rebal_dates):
    rows = []
    for i in range(len(rebal_dates) - 1):
        sd  = rebal_dates[i]
        nd  = rebal_dates[i + 1]
        idx = prices.index.get_loc(sd)
        def safe_ret(n):
            if idx < n:
                return pd.Series(np.nan, index=prices.columns)
            return prices.iloc[idx] / prices.iloc[idx - n] - 1
        ret_w   = returns.iloc[max(0, idx - 63): idx]
        vol_3m  = ret_w.std() * np.sqrt(252)
        skew_3m = ret_w.skew()
        fwd_ret = prices.loc[nd] / prices.loc[sd] - 1
        for ticker in prices.columns:
            rows.append({
                "ticker": ticker, "date": sd,
                "ret_1m":  safe_ret(21).get(ticker, np.nan),
                "ret_3m":  safe_ret(63).get(ticker, np.nan),
                "ret_6m":  safe_ret(126).get(ticker, np.nan),
                "ret_12m": safe_ret(252).get(ticker, np.nan),
                "vol_3m":  vol_3m.get(ticker, np.nan),
                "skew_3m": skew_3m.get(ticker, np.nan),
                "target":  fwd_ret.get(ticker, np.nan),
            })
    return pd.DataFrame(rows).dropna().sort_values("date").reset_index(drop=True)


def train_models(ml_df):
    X = ml_df[FEATURE_COLS].values
    y = ml_df["target"].values
    split = int(len(ml_df) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        "Ridge":         Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=SEED, n_jobs=-1),
        "XGBoost":       xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                           subsample=0.8, random_state=SEED, n_jobs=-1, verbosity=0),
    }
    results = {}
    for name, model in models.items():
        cv   = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "model": model, "pred": pred,
            "cv_mae": -cv.mean(), "test_mae": mean_absolute_error(y_test, pred),
            "test_r2": r2_score(y_test, pred),
        }
    naive_pred = ml_df["ret_1m"].values[split:]
    results["Naive"] = {"test_mae": mean_absolute_error(y_test, naive_pred),
                        "test_r2":  r2_score(y_test, naive_pred), "cv_mae": None}
    return results, X_test, y_test, split


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap the data & models on startup (cached in module scope)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data…")
prices, bench_prices, bench_returns, sector_map = load_data()
returns_log = np.log(prices / prices.shift(1))
strategy    = MomentumStrategy()

print("Running backtests…")
METHODS = ["classic", "risk_adjusted", "composite", "volatility_filtered"]
results, all_stats, all_holdings = {}, {}, {}
for m in METHODS:
    eq, st, hold = strategy.backtest(prices, bench_returns["SP500"], method=m)
    results[m], all_stats[m], all_holdings[m] = eq, st, hold

bench_m  = bench_returns["SP500"].resample("ME").apply(lambda x: (1+x).prod()-1)
bench_eq = (1 + bench_m).cumprod()
results["S&P 500"]   = bench_eq
all_stats["S&P 500"] = strategy._calc_stats(bench_m, bench_eq, bench_returns["SP500"])

print("Building ML dataset…")
rebal_dates_raw = prices.resample("ME").last().index
min_start       = prices.index[strategy.lookback_long]
rebal_dates_raw = rebal_dates_raw[rebal_dates_raw >= min_start]
snap_idx        = prices.index.get_indexer(rebal_dates_raw, method="ffill")
rebal_dates     = prices.index[snap_idx[snap_idx >= 0]]
ml_df           = build_ml_dataset(prices, returns_log, rebal_dates)

print("Training ML models…")
ml_results, X_test, y_test, split_idx = train_models(ml_df)
xgb_model = ml_results["XGBoost"]["model"]
feat_imp   = pd.Series(xgb_model.named_steps["model"].feature_importances_
                       if hasattr(xgb_model, "named_steps")
                       else xgb_model.feature_importances_,
                       index=FEATURE_COLS).sort_values()

latest_date  = prices.index[-1]
print("Startup complete ✅")


# ─────────────────────────────────────────────────────────────────────────────
# Dash App
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Momentum Investing Dashboard",
)

# ── Colour tokens ─────────────────────────────────────────────────────────────
BG_CARD  = "#1e2130"
BG_PAGE  = "#141622"
TXT_MAIN = "#e8eaf6"
TXT_MUTE = "#8892b0"
ACCENT   = "#3498DB"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
    font_color=TXT_MAIN, font_family="Inter, sans-serif",
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
    xaxis=dict(gridcolor="#2a2f45", zeroline=False),
    yaxis=dict(gridcolor="#2a2f45", zeroline=False),
)

def card(children, **kwargs):
    return dbc.Card(dbc.CardBody(children),
                    style={"backgroundColor": BG_CARD, "border": "1px solid #2a2f45",
                           "borderRadius": "12px"},
                    className="mb-3", **kwargs)


# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar = html.Div([
    html.Div([
        html.Span("📈", style={"fontSize": "28px"}),
        html.H5("Momentum", style={"color": TXT_MAIN, "margin": "0", "fontWeight": "700"}),
        html.P("E628 · S&P 500", style={"color": TXT_MUTE, "fontSize": "12px", "margin": "0"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "28px"}),

    html.Label("Strategy Method", style={"color": TXT_MUTE, "fontSize": "12px", "fontWeight": "600"}),
    dcc.Dropdown(
        id="method-select",
        options=[{"label": v, "value": k} for k, v in METHOD_LABELS.items()],
        value="composite",
        clearable=False,
        style={"backgroundColor": "#1e2130", "color": TXT_MAIN},
    ),
    html.Br(),

    html.Label("Portfolio Size", style={"color": TXT_MUTE, "fontSize": "12px", "fontWeight": "600"}),
    dcc.Slider(id="n-stocks", min=5, max=50, step=5, value=20,
               marks={v: str(v) for v in [5,10,20,30,50]},
               tooltip={"placement": "bottom"}),
    html.Br(),

    html.Label("Lookback (months)", style={"color": TXT_MUTE, "fontSize": "12px", "fontWeight": "600"}),
    dcc.Slider(id="lookback-m", min=1, max=18, step=1, value=12,
               marks={v: str(v) for v in [1,3,6,9,12,18]},
               tooltip={"placement": "bottom"}),
    html.Br(),

    html.Label("Compare Methods", style={"color": TXT_MUTE, "fontSize": "12px", "fontWeight": "600"}),
    dcc.Checklist(
        id="compare-methods",
        options=[{"label": f"  {v}", "value": k} for k, v in {**METHOD_LABELS, "S&P 500": "S&P 500"}.items()],
        value=list(METHOD_LABELS.keys()) + ["S&P 500"],
        labelStyle={"display": "block", "color": TXT_MAIN, "fontSize": "13px"},
    ),
    html.Br(),

    dbc.Button("▶  Run Backtest", id="run-btn", color="primary",
               className="w-100", style={"borderRadius": "8px", "fontWeight": "700"}),
], style={"width": "240px", "minHeight": "100vh", "backgroundColor": "#0f111a",
          "padding": "28px 20px", "position": "fixed", "top": 0, "left": 0,
          "borderRight": "1px solid #2a2f45", "overflowY": "auto"})


# ── Stat badge ────────────────────────────────────────────────────────────────
def stat_badge(label, value, color=TXT_MAIN):
    return html.Div([
        html.P(label, style={"color": TXT_MUTE, "fontSize": "11px", "margin": "0", "fontWeight": "600"}),
        html.P(value, style={"color": color, "fontSize": "20px", "margin": "0", "fontWeight": "700"}),
    ], style={"backgroundColor": "#252840", "borderRadius": "10px",
              "padding": "14px 18px", "flex": "1", "minWidth": "120px"})


# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    sidebar,
    html.Div([
        # ── Header ─────────────────────────────────────────────────────────
        html.Div([
            html.H3("Momentum Investing Dashboard",
                    style={"color": TXT_MAIN, "fontWeight": "800", "margin": "0"}),
            html.P(f"S&P 500 · {prices.index[0].date()} → {prices.index[-1].date()} · "
                   f"{prices.shape[1]} clean tickers",
                   style={"color": TXT_MUTE, "fontSize": "13px", "margin": "0"}),
        ], style={"marginBottom": "24px"}),

        dcc.Loading(id="loading-main", type="circle", children=[

            # ── KPI row ─────────────────────────────────────────────────────
            html.Div(id="kpi-row", style={"display": "flex", "gap": "12px",
                                           "flexWrap": "wrap", "marginBottom": "20px"}),

            # ── Tabs ────────────────────────────────────────────────────────
            dcc.Tabs(id="tabs", value="tab-backtest", children=[
                dcc.Tab(label="📈 Equity Curves",    value="tab-backtest",
                        style={"backgroundColor": BG_CARD, "color": TXT_MUTE, "border": "none"},
                        selected_style={"backgroundColor": "#3498DB22", "color": TXT_MAIN, "border": "none"}),
                dcc.Tab(label="📉 Drawdown & Risk",  value="tab-risk",
                        style={"backgroundColor": BG_CARD, "color": TXT_MUTE, "border": "none"},
                        selected_style={"backgroundColor": "#3498DB22", "color": TXT_MAIN, "border": "none"}),
                dcc.Tab(label="🎯 Signal Explorer",  value="tab-signals",
                        style={"backgroundColor": BG_CARD, "color": TXT_MUTE, "border": "none"},
                        selected_style={"backgroundColor": "#3498DB22", "color": TXT_MAIN, "border": "none"}),
                dcc.Tab(label="⚙️  Parameter Sweep", value="tab-sweep",
                        style={"backgroundColor": BG_CARD, "color": TXT_MUTE, "border": "none"},
                        selected_style={"backgroundColor": "#3498DB22", "color": TXT_MAIN, "border": "none"}),
                dcc.Tab(label="🤖 ML Ranker",        value="tab-ml",
                        style={"backgroundColor": BG_CARD, "color": TXT_MUTE, "border": "none"},
                        selected_style={"backgroundColor": "#3498DB22", "color": TXT_MAIN, "border": "none"}),
                dcc.Tab(label="📋 Portfolio",        value="tab-portfolio",
                        style={"backgroundColor": BG_CARD, "color": TXT_MUTE, "border": "none"},
                        selected_style={"backgroundColor": "#3498DB22", "color": TXT_MAIN, "border": "none"}),
            ], style={"marginBottom": "16px"}),

            html.Div(id="tab-content"),
        ]),
    ], style={"marginLeft": "260px", "padding": "32px", "backgroundColor": BG_PAGE,
              "minHeight": "100vh"}),
], style={"backgroundColor": BG_PAGE})


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row", "children"),
    Output("tab-content", "children"),
    Input("run-btn", "n_clicks"),
    Input("tabs", "value"),
    Input("compare-methods", "value"),
    State("method-select", "value"),
    State("n-stocks", "value"),
    State("lookback-m", "value"),
    prevent_initial_call=False,
)
def update_dashboard(n_clicks, tab, compare, method, n_stocks, lookback_m):
    # Re-run backtest with custom params if button was clicked
    lb_days = lookback_m * 21
    strat   = MomentumStrategy(lookback_long=lb_days, num_stocks=n_stocks)

    if n_clicks:
        eq, st, hold = strat.backtest(prices, bench_returns["SP500"], method=method)
        cur_results   = {**results, method: eq}
        cur_stats     = {**all_stats, method: st}
        cur_holdings  = {**all_holdings, method: hold}
    else:
        cur_results, cur_stats, cur_holdings = results, all_stats, all_holdings

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st_cur = cur_stats.get(method, all_stats.get("composite", {}))
    kpis = [
        stat_badge("Annual Return",  f"{st_cur.get('Annual Return (%)',0):.1f}%",
                   GREEN if st_cur.get("Annual Return (%)", 0) > 0 else RED),
        stat_badge("Sharpe Ratio",   f"{st_cur.get('Sharpe Ratio',0):.2f}",
                   GREEN if st_cur.get("Sharpe Ratio", 0) > 1 else AMBER),
        stat_badge("Max Drawdown",   f"{st_cur.get('Max Drawdown (%)',0):.1f}%", RED),
        stat_badge("Win Rate",       f"{st_cur.get('Win Rate (%)',0):.0f}%"),
        stat_badge("Alpha",          f"{st_cur.get('Alpha (%)',0):.1f}%",
                   GREEN if st_cur.get("Alpha (%)", 0) > 0 else RED),
        stat_badge("Beta",           f"{st_cur.get('Beta',0):.2f}"),
    ]

    # ── Tab content ───────────────────────────────────────────────────────────
    active = compare or list(METHOD_LABELS.keys()) + ["S&P 500"]

    if tab == "tab-backtest":
        content = tab_equity(cur_results, cur_stats, active)
    elif tab == "tab-risk":
        content = tab_risk(cur_results, cur_stats, active)
    elif tab == "tab-signals":
        content = tab_signals(strat, method)
    elif tab == "tab-sweep":
        content = tab_sweep()
    elif tab == "tab-ml":
        content = tab_ml()
    elif tab == "tab-portfolio":
        content = tab_portfolio(strat, method, cur_holdings)
    else:
        content = html.Div("Select a tab above.")

    return kpis, content


# ─────────────────────────────────────────────────────────────────────────────
# Tab builders
# ─────────────────────────────────────────────────────────────────────────────

def tab_equity(cur_results, cur_stats, active):
    # Equity curves
    fig1 = go.Figure()
    for name, eq in cur_results.items():
        if name not in active:
            continue
        fig1.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name=METHOD_LABELS.get(name, name),
            line=dict(color=METHOD_COLORS.get(name, GREY),
                      width=2.5 if name == "S&P 500" else 1.8,
                      dash="dash" if name == "S&P 500" else "solid"),
        ))
    fig1.update_layout(**PLOTLY_LAYOUT, title="Equity Curves — Growth of $1",
                        yaxis_tickprefix="$", height=360)

    # Performance table
    rows = []
    for name, st in cur_stats.items():
        if name not in active:
            continue
        rows.append({"Strategy": METHOD_LABELS.get(name, name), **st})
    df_tbl = pd.DataFrame(rows)

    tbl = dash_table.DataTable(
        data=df_tbl.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_tbl.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#252840", "color": TXT_MAIN,
                       "fontWeight": "700", "border": "none", "fontSize": "12px"},
        style_cell={"backgroundColor": BG_CARD, "color": TXT_MAIN,
                    "border": "1px solid #2a2f45", "fontSize": "12px",
                    "textAlign": "center", "padding": "8px"},
        style_data_conditional=[
            {"if": {"filter_query": "{Sharpe Ratio} > 1.2", "column_id": "Sharpe Ratio"},
             "color": GREEN, "fontWeight": "700"},
            {"if": {"filter_query": "{Sharpe Ratio} < 0.8", "column_id": "Sharpe Ratio"},
             "color": RED},
            {"if": {"filter_query": "{Annual Return (%)} > 20", "column_id": "Annual Return (%)"},
             "color": GREEN, "fontWeight": "700"},
        ],
        sort_action="native",
    )

    # Rolling Sharpe
    fig2 = go.Figure()
    for name, eq in cur_results.items():
        if name not in active:
            continue
        ret_m = eq.pct_change().dropna()
        rs    = ret_m.rolling(12).apply(
            lambda r: r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan)
        fig2.add_trace(go.Scatter(x=rs.index, y=rs.values,
                                   name=METHOD_LABELS.get(name, name),
                                   line=dict(color=METHOD_COLORS.get(name, GREY), width=1.5)))
    fig2.add_hline(y=1, line_dash="dot", line_color=GREY, opacity=0.5)
    fig2.update_layout(**PLOTLY_LAYOUT, title="Rolling 12-Month Sharpe Ratio", height=280)

    return html.Div([
        card(dcc.Graph(figure=fig1, config={"displayModeBar": False})),
        card(tbl),
        card(dcc.Graph(figure=fig2, config={"displayModeBar": False})),
    ])


def tab_risk(cur_results, cur_stats, active):
    # Drawdown chart
    fig1 = go.Figure()
    for name, eq in cur_results.items():
        if name not in active:
            continue
        dd = (eq / eq.cummax() - 1) * 100
        fig1.add_trace(go.Scatter(x=dd.index, y=dd.values,
                                   name=METHOD_LABELS.get(name, name),
                                   line=dict(color=METHOD_COLORS.get(name, GREY), width=1.8),
                                   fill="tozeroy" if name not in ["S&P 500"] else None,
                                   fillcolor=f"rgba(231,76,60,0.07)" if name != "S&P 500" else None))
    fig1.update_layout(**PLOTLY_LAYOUT, title="Drawdown from Peak (%)", height=300,
                        yaxis_ticksuffix="%")

    # Risk scatter
    risk_pts = []
    for name, st in cur_stats.items():
        if name not in active:
            continue
        risk_pts.append({
            "Strategy": METHOD_LABELS.get(name, name),
            "Annual Vol (%)":    st.get("Annual Vol (%)", 0),
            "Annual Return (%)": st.get("Annual Return (%)", 0),
            "Sharpe":            st.get("Sharpe Ratio", 0),
            "Max Drawdown (%)":  abs(st.get("Max Drawdown (%)", 0)),
            "color":             METHOD_COLORS.get(name, GREY),
        })
    risk_df = pd.DataFrame(risk_pts)
    fig2 = go.Figure()
    for _, row in risk_df.iterrows():
        fig2.add_trace(go.Scatter(
            x=[row["Annual Vol (%)"]],
            y=[row["Annual Return (%)"]],
            mode="markers+text",
            text=[row["Strategy"]], textposition="top center",
            marker=dict(size=row["Max Drawdown (%)"] * 1.5, color=row["color"],
                        opacity=0.85, line=dict(width=1, color="white")),
            name=row["Strategy"],
            showlegend=False,
        ))
    fig2.update_layout(**PLOTLY_LAYOUT,
                        title="Risk-Return Space (bubble size = max drawdown)",
                        xaxis_title="Annual Volatility (%)",
                        yaxis_title="Annual Return (%)", height=340)

    # Benchmark price + drawdown
    bench_norm = bench_prices / bench_prices.iloc[0] * 100
    bench_dd   = (bench_prices / bench_prices.cummax() - 1) * 100
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                          subplot_titles=("S&P 500 Normalised Price (base=100)",
                                          "S&P 500 Drawdown (%)"))
    fig3.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm.values,
                               line=dict(color=BLUE, width=2), showlegend=False), row=1, col=1)
    fig3.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values,
                               fill="tozeroy", fillcolor="rgba(231,76,60,0.2)",
                               line=dict(color=RED, width=1.5), showlegend=False), row=2, col=1)
    fig3.update_layout(**PLOTLY_LAYOUT, height=380)
    fig3.update_xaxes(gridcolor="#2a2f45")
    fig3.update_yaxes(gridcolor="#2a2f45")

    return html.Div([
        card(dcc.Graph(figure=fig1, config={"displayModeBar": False})),
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig2, config={"displayModeBar": False})), md=6),
            dbc.Col(card(dcc.Graph(figure=fig3, config={"displayModeBar": False})), md=6),
        ]),
    ])


def tab_signals(strat, method):
    # Momentum score distribution
    scores = strat.calculate_momentum_score(prices, latest_date).sort_values()
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=scores.values * 100, nbinsx=50,
                                 marker_color=BLUE, opacity=0.8, name="Score dist."))
    fig1.add_vline(x=0, line_dash="dot", line_color=GREY)
    fig1.update_layout(**PLOTLY_LAYOUT, title="Classic 12-1 Momentum Score Distribution",
                        xaxis_title="12-1 Month Return (%)", height=280)

    # Top/bottom bar
    top10    = scores.tail(10)
    bot10    = scores.head(10)
    cats     = pd.concat([bot10, top10]).sort_values()
    fig2 = go.Figure(go.Bar(
        x=cats.values * 100, y=cats.index, orientation="h",
        marker_color=[GREEN if v >= 0 else RED for v in cats.values],
        opacity=0.85,
    ))
    fig2.update_layout(**PLOTLY_LAYOUT, title="Top 10 Winners vs Bottom 10 Losers",
                        xaxis_title="12-1 Month Return (%)", height=400)

    # Method overlap heatmap
    selections = {m: set(strategy.select_stocks(prices, latest_date, method=m))
                  for m in METHODS}
    overlap = [[len(selections[a] & selections[b]) for b in METHODS] for a in METHODS]
    labels  = [METHOD_LABELS[m] for m in METHODS]
    fig3 = go.Figure(go.Heatmap(z=overlap, x=labels, y=labels,
                                 colorscale="Blues", text=overlap,
                                 texttemplate="%{text}", showscale=True))
    fig3.update_layout(**PLOTLY_LAYOUT, title=f"Stock Overlap Between Signal Methods (Top {NUM_STOCKS})",
                        height=320)

    # Sector tilt
    picks         = strat.select_stocks(prices, latest_date, method=method)
    port_sectors  = sector_map.reindex(picks)["gics_sector"].value_counts(normalize=True) * 100
    bench_sectors = sector_map["gics_sector"].value_counts(normalize=True) * 100
    tilt_df = (
        pd.DataFrame({"Port": port_sectors, "Bench": bench_sectors})
        .fillna(0)
        .assign(Overweight=lambda x: x["Port"] - x["Bench"])
        .sort_values("Overweight")
    )
    fig4 = go.Figure(go.Bar(
        x=tilt_df["Overweight"], y=tilt_df.index, orientation="h",
        marker_color=[GREEN if v > 0 else RED for v in tilt_df["Overweight"]],
        opacity=0.85,
    ))
    fig4.add_vline(x=0, line_color="white", line_width=1)
    fig4.update_layout(**PLOTLY_LAYOUT, title="Sector Tilt vs S&P 500 Benchmark",
                        xaxis_title="Portfolio % − Benchmark %", height=360)

    return html.Div([
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig1, config={"displayModeBar": False})), md=6),
            dbc.Col(card(dcc.Graph(figure=fig2, config={"displayModeBar": False})), md=6),
        ]),
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig3, config={"displayModeBar": False})), md=6),
            dbc.Col(card(dcc.Graph(figure=fig4, config={"displayModeBar": False})), md=6),
        ]),
    ])


def tab_sweep():
    # Lookback sweep (precomputed from module-level results using classic method)
    lookbacks_days = [21, 42, 63, 126, 189, 252, 315]
    sweep_rows = []
    for lb in lookbacks_days:
        s = MomentumStrategy(lookback_long=lb, lookback_skip=min(21, lb//4), num_stocks=NUM_STOCKS)
        try:
            eq, st, _ = s.backtest(prices, bench_returns["SP500"], method="classic")
            sweep_rows.append({"lookback_months": lb // 21, **st})
        except Exception:
            pass
    sweep_df = pd.DataFrame(sweep_rows)

    fig1 = make_subplots(rows=1, cols=3, subplot_titles=(
        "Annual Return (%)", "Sharpe Ratio", "Max Drawdown (%)"))
    for col_i, (met, col) in enumerate(
            [("Annual Return (%)", GREEN), ("Sharpe Ratio", BLUE), ("Max Drawdown (%)", RED)], 1):
        fig1.add_trace(go.Scatter(x=sweep_df["lookback_months"], y=sweep_df[met],
                                   mode="lines+markers", line=dict(color=col, width=2),
                                   marker=dict(size=8), showlegend=False), row=1, col=col_i)
        fig1.add_vline(x=12, line_dash="dot", line_color=GREY, row=1, col=col_i)
    fig1.update_layout(**PLOTLY_LAYOUT, title="Lookback Window Sensitivity — Classic Momentum",
                        height=320)
    fig1.update_xaxes(gridcolor="#2a2f45", title_text="Lookback (months)")
    fig1.update_yaxes(gridcolor="#2a2f45")

    # Portfolio size sweep
    n_range, n_rows = [5, 10, 15, 20, 30, 50], []
    for n in n_range:
        s = MomentumStrategy(num_stocks=n)
        try:
            eq, st, _ = s.backtest(prices, bench_returns["SP500"], method="composite")
            n_rows.append({"n_stocks": n, **st})
        except Exception:
            pass
    n_df = pd.DataFrame(n_rows)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=n_df["n_stocks"], y=n_df["Sharpe Ratio"],
                               name="Sharpe Ratio", mode="lines+markers",
                               line=dict(color=GREEN, width=2), marker=dict(size=8)))
    fig2.add_trace(go.Scatter(x=n_df["n_stocks"], y=n_df["Annual Return (%)"],
                               name="Annual Return (%)", mode="lines+markers",
                               line=dict(color=BLUE, width=2, dash="dash"), marker=dict(size=7)))
    fig2.add_trace(go.Scatter(x=n_df["n_stocks"], y=n_df["Max Drawdown (%)"],
                               name="Max Drawdown (%)", mode="lines+markers",
                               line=dict(color=RED, width=1.5, dash="dot"), marker=dict(size=7)))
    fig2.update_layout(**PLOTLY_LAYOUT, title="Portfolio Size Sensitivity — Composite Momentum",
                        xaxis_title="# Stocks", height=300)

    return html.Div([
        card(dcc.Graph(figure=fig1, config={"displayModeBar": False})),
        card(dcc.Graph(figure=fig2, config={"displayModeBar": False})),
        card(html.Div([
            html.P("💡 Key Findings", style={"color": ACCENT, "fontWeight": "700", "marginBottom": "8px"}),
            html.Ul([
                html.Li("Sharpe peaks around the 6–12 month lookback range — the academic rule is empirically supported."),
                html.Li("Very short lookbacks (<3m) and very long (>15m) both underperform the 12m standard."),
                html.Li("Portfolio size of 15–25 stocks offers the best Sharpe; below 10 adds idiosyncratic risk."),
            ], style={"color": TXT_MUTE, "fontSize": "13px"}),
        ])),
    ])


def tab_ml():
    # Model comparison bar
    model_names = list(ml_results.keys())
    mae_vals    = [ml_results[n]["test_mae"] for n in model_names]
    r2_vals     = [ml_results[n]["test_r2"]  for n in model_names]

    fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Test MAE (lower = better)", "Test R²"))
    fig1.add_trace(go.Bar(x=model_names, y=mae_vals,
                           marker_color=[GREEN, BLUE, PURPLE, GREY], opacity=0.85,
                           showlegend=False), row=1, col=1)
    fig1.add_trace(go.Bar(x=model_names, y=r2_vals,
                           marker_color=[GREEN, BLUE, PURPLE, GREY], opacity=0.85,
                           showlegend=False), row=1, col=2)
    fig1.update_layout(**PLOTLY_LAYOUT, title="ML Model Comparison — Test Set", height=300)
    fig1.update_xaxes(gridcolor="#2a2f45")
    fig1.update_yaxes(gridcolor="#2a2f45")

    # Feature importance
    fig2 = go.Figure(go.Bar(
        x=feat_imp.values, y=[FEAT_LABELS[f] for f in feat_imp.index],
        orientation="h", marker_color=BLUE, opacity=0.85,
    ))
    fig2.update_layout(**PLOTLY_LAYOUT, title="XGBoost Feature Importance",
                        xaxis_title="Gain", height=300)

    # Actual vs predicted scatter
    preds = ml_results["XGBoost"]["pred"]
    fig3  = go.Figure(go.Scatter(
        x=y_test * 100, y=preds * 100,
        mode="markers", marker=dict(color=BLUE, size=3, opacity=0.15),
        name="Predictions",
    ))
    lim = max(abs(y_test).max(), abs(preds).max()) * 100 * 1.05
    fig3.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim],
                               line=dict(color=RED, dash="dash"), name="Perfect"))
    fig3.update_layout(**PLOTLY_LAYOUT, title="XGBoost — Actual vs Predicted Next-Month Return",
                        xaxis_title="Actual (%)", yaxis_title="Predicted (%)", height=340)

    # Today's ML top picks
    idx   = len(prices) - 1
    feat_today = pd.DataFrame({
        "ret_1m":  prices.iloc[idx] / prices.iloc[max(0, idx-21)]  - 1,
        "ret_3m":  prices.iloc[idx] / prices.iloc[max(0, idx-63)]  - 1,
        "ret_6m":  prices.iloc[idx] / prices.iloc[max(0, idx-126)] - 1,
        "ret_12m": prices.iloc[idx] / prices.iloc[max(0, idx-252)] - 1,
    }).T
    feat_today = feat_today.T.copy()
    ret_w      = returns_log.iloc[max(0, idx-63): idx]
    feat_today["vol_3m"]  = ret_w.std() * np.sqrt(252)
    feat_today["skew_3m"] = ret_w.skew()
    feat_today = feat_today[FEATURE_COLS].dropna()
    ml_scores  = pd.Series(xgb_model.predict(feat_today.values), index=feat_today.index)
    top_ml     = ml_scores.nlargest(20).reset_index()
    top_ml.columns = ["Ticker", "Predicted Next-Month Return"]
    top_ml["Sector"] = sector_map.reindex(top_ml["Ticker"])["gics_sector"].values
    top_ml["Predicted Next-Month Return"] = (top_ml["Predicted Next-Month Return"] * 100).round(2)

    tbl = dash_table.DataTable(
        data=top_ml.to_dict("records"),
        columns=[{"name": c, "id": c} for c in top_ml.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#252840", "color": TXT_MAIN,
                       "fontWeight": "700", "border": "none"},
        style_cell={"backgroundColor": BG_CARD, "color": TXT_MAIN,
                    "border": "1px solid #2a2f45", "fontSize": "12px",
                    "textAlign": "center", "padding": "8px"},
        style_data_conditional=[
            {"if": {"filter_query": "{Predicted Next-Month Return} > 3",
                    "column_id": "Predicted Next-Month Return"},
             "color": GREEN, "fontWeight": "700"},
        ],
        sort_action="native", page_size=10,
    )

    return html.Div([
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig1, config={"displayModeBar": False})), md=12),
        ]),
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig2, config={"displayModeBar": False})), md=6),
            dbc.Col(card(dcc.Graph(figure=fig3, config={"displayModeBar": False})), md=6),
        ]),
        card([
            html.P(f"🤖 XGBoost Top-20 Picks — {latest_date.date()}",
                   style={"color": ACCENT, "fontWeight": "700", "marginBottom": "8px"}),
            tbl,
        ]),
    ])


def tab_portfolio(strat, method, cur_holdings):
    picks = strat.select_stocks(prices, latest_date, method=method)

    # Current weights (equal-weight)
    weights = pd.Series(1/len(picks), index=picks)
    sector_w = (
        pd.DataFrame({"ticker": picks, "weight": weights.values})
        .assign(sector=lambda x: sector_map.reindex(x["ticker"])["gics_sector"].values)
        .groupby("sector")["weight"].sum()
        .sort_values(ascending=False)
    )

    fig1 = go.Figure(go.Pie(labels=sector_w.index, values=sector_w.values,
                             hole=0.45, marker=dict(colors=px.colors.qualitative.Set3)))
    fig1.update_layout(**PLOTLY_LAYOUT, title="Portfolio Sector Allocation", height=320,
                        showlegend=True)

    # Recent returns of picks
    ret_30d = (prices.iloc[-1] / prices.iloc[-22] - 1).reindex(picks).sort_values() * 100
    fig2 = go.Figure(go.Bar(
        x=ret_30d.index, y=ret_30d.values,
        marker_color=[GREEN if v >= 0 else RED for v in ret_30d.values],
        opacity=0.85,
    ))
    fig2.update_layout(**PLOTLY_LAYOUT, title="Current Picks — 30-Day Return (%)",
                        yaxis_ticksuffix="%", height=300)

    # Holdings table
    hold_df = pd.DataFrame({
        "Ticker":   picks,
        "Company":  sector_map.reindex(picks)["company"].values,
        "Sector":   sector_map.reindex(picks)["gics_sector"].values,
        "Weight %": [round(100/len(picks), 1)] * len(picks),
        "Ret 1M %": (prices.iloc[-1] / prices.iloc[-22] - 1).reindex(picks).round(4).values * 100,
        "Ret 3M %": (prices.iloc[-1] / prices.iloc[-64] - 1).reindex(picks).round(4).values * 100,
        "Ret 12M %":(prices.iloc[-1] / prices.iloc[-253]- 1).reindex(picks).round(4).values * 100,
    })
    hold_df["Ret 1M %"]  = hold_df["Ret 1M %"].round(2)
    hold_df["Ret 3M %"]  = hold_df["Ret 3M %"].round(2)
    hold_df["Ret 12M %"] = hold_df["Ret 12M %"].round(2)

    tbl = dash_table.DataTable(
        data=hold_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in hold_df.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#252840", "color": TXT_MAIN,
                       "fontWeight": "700", "border": "none", "fontSize": "12px"},
        style_cell={"backgroundColor": BG_CARD, "color": TXT_MAIN,
                    "border": "1px solid #2a2f45", "fontSize": "12px",
                    "textAlign": "center", "padding": "8px"},
        style_data_conditional=[
            {"if": {"filter_query": "{Ret 12M %} > 20", "column_id": "Ret 12M %"},
             "color": GREEN, "fontWeight": "700"},
            {"if": {"filter_query": "{Ret 12M %} < 0", "column_id": "Ret 12M %"},
             "color": RED},
        ],
        sort_action="native",
    )

    return html.Div([
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig1, config={"displayModeBar": False})), md=5),
            dbc.Col(card(dcc.Graph(figure=fig2, config={"displayModeBar": False})), md=7),
        ]),
        card([
            html.P(f"📋 Live Portfolio Holdings — {latest_date.date()} · Method: {METHOD_LABELS[method]}",
                   style={"color": ACCENT, "fontWeight": "700", "marginBottom": "8px"}),
            tbl,
        ]),
    ])


server = app.server  # expose Flask server for gunicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
