"""Weekly ML retrain for Signal Desk.

1. Downloads as much price history as Yahoo allows for your candle size.
2. Replays the bot's own rules over it to find every BUY/SELL switch.
3. Labels each one: did it make money after Kraken fees within the holding time?
4. Trains a gradient-boosting model (LightGBM) to score signals.
5. Tests it walk-forward: train on the past, test on the weeks after, never the
   other way round. The model is switched on only if the signals it likes beat
   the plain rules out-of-sample, in most test periods.

Writes data/ml_model.txt and data/ml_report.json, and sends a summary to your phone.
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from features import FEATURES, Indicators, features_at, signal_events

DATA = Path("data")
MODEL_FILE = DATA / "ml_model.txt"
REPORT_FILE = DATA / "ml_report.json"
INTERVAL_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
# Yahoo: (interval to download, how far back it allows)
YAHOO = {"1m": ("1m", "7d"), "5m": ("5m", "60d"), "15m": ("15m", "60d"),
         "1h": ("1h", "730d"), "4h": ("1h", "730d"), "1d": ("1d", "max")}
LGB_PARAMS = dict(n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def env(name, default=""):
    v = os.environ.get(name, "")
    return v.strip() if v and v.strip() else default


INTERVAL = env("INTERVAL", "15m")
FEE = float(env("ML_FEE_PCT", "0.8")) / 100           # Kraken Pro round trip
HOLD_HOURS = float(env("ML_HOLD_HOURS", "24"))
COINS = sorted({c.strip().upper() for c in
                (env("COINS", "BTC,ETH,SOL") + "," +
                 env("ML_TRAIN_COINS", "BTC,ETH,SOL,XRP,DOGE,ADA,AVAX,LINK,LTC,DOT")).split(",")
                if c.strip()})
FOLDS = 5
MIN_SIGNALS = 300


# ---------------------------------------------------------------- data
def yahoo_history(coin):
    """Closed candles as dict of lists: t, h, l, c, v (UTC seconds)."""
    import yfinance as yf

    ticker = {"XBT": "BTC", "XDG": "DOGE"}.get(coin, coin) + "-USD"
    overrides = dict(p.split("=", 1) for p in env("YAHOO_TICKERS").split(",") if "=" in p)
    ticker = overrides.get(coin, ticker)
    interval, period = YAHOO[INTERVAL]
    for attempt in range(3):
        try:
            df = yf.download(ticker, interval=interval, period=period, progress=False,
                             auto_adjust=False, multi_level_index=False)
            break
        except Exception as e:
            print(f"{ticker}: attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    else:
        return None
    if df is None or df.empty:
        return None
    df = df.dropna()
    df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    if INTERVAL == "4h":
        df = df.resample("4h", origin="epoch").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    df = df.iloc[:-1]  # last candle may still be forming
    return {"t": [int(x.timestamp()) for x in df.index], "h": df["High"].astype(float).tolist(),
            "l": df["Low"].astype(float).tolist(), "c": df["Close"].astype(float).tolist(),
            "v": df["Volume"].astype(float).tolist()}


def build_dataset(histories):
    """One row per historical BUY/SELL switch: features, label, net return, time."""
    hold = max(1, round(HOLD_HOURS * 60 / INTERVAL_MIN[INTERVAL]))
    X, y, net, times, coins = [], [], [], [], []
    for coin, h in histories.items():
        ind = Indicators(h["c"])
        for i, sig in signal_events(ind):
            if i < 100 or i + hold >= len(h["c"]):
                continue  # need warm-up before, and a known outcome after
            side = 1 if sig == "BUY" else -1
            r = side * (h["c"][i + hold] / h["c"][i] - 1) - FEE
            X.append(features_at(ind, i, side, h["t"], h["h"], h["l"], h["v"]))
            y.append(1 if r > 0 else 0)
            net.append(r)
            times.append(h["t"][i])
            coins.append(coin)
    order = np.argsort(times, kind="stable")
    return (np.array(X, float)[order], np.array(y)[order], np.array(net)[order],
            np.array(times)[order], np.array(coins)[order], hold)


# ---------------------------------------------------------------- model + walk-forward
def new_model():
    import lightgbm as lgb
    return lgb.LGBMClassifier(**LGB_PARAMS)


def walk_forward(X, y, net, times, hold):
    """Train on everything before each test block, test on the block. Returns per-fold results."""
    embargo = hold * INTERVAL_MIN[INTERVAL] * 60
    edges = np.linspace(0, len(X), FOLDS + 1).astype(int)
    folds, oos = [], []
    for k in range(1, FOLDS):
        lo, hi = edges[k], edges[k + 1]
        test_start = times[lo]
        # drop training signals whose outcome window reaches into the test block
        train = np.where(times < test_start - embargo)[0]
        if len(train) < 100 or hi - lo < 20:
            continue
        m = new_model().fit(X[train], y[train])
        p = m.predict_proba(X[lo:hi])[:, 1]
        cutoff = y[train].mean()  # "better than this model's average odds"
        picked = p > cutoff
        f = {"from": datetime.fromtimestamp(times[lo], timezone.utc).strftime("%Y-%m-%d"),
             "to": datetime.fromtimestamp(times[hi - 1], timezone.utc).strftime("%Y-%m-%d"),
             "signals": int(hi - lo), "picked": int(picked.sum()),
             "rules_avg": float(net[lo:hi].mean()), "rules_win": float(y[lo:hi].mean()),
             "ml_avg": float(net[lo:hi][picked].mean()) if picked.any() else None,
             "ml_win": float(y[lo:hi][picked].mean()) if picked.any() else None}
        f["ml_better"] = f["ml_avg"] is not None and f["ml_avg"] > f["rules_avg"]
        folds.append(f)
        oos.append((net[lo:hi], picked))
    return folds, oos


def judge(folds, oos):
    """Switch ML on only if it clearly helped out-of-sample.

    All of these must hold on the test periods (data the model never trained on):
    - its picked trades made money on average after fees,
    - picked trades beat skipped trades by a statistically clear margin (t > 2),
    - it beat the plain rules in all but at most one test period.
    """
    if len(folds) < 3:
        return False, "not enough history to test it"
    nets = np.concatenate([n for n, _ in oos])
    mask = np.concatenate([p for _, p in oos])
    picked, skipped = nets[mask], nets[~mask]
    rules = nets.mean()
    if len(picked) < 30 or len(skipped) < 30:
        return False, "it picked too few (or too many) signals to judge"
    ml = picked.mean()
    se = np.sqrt(picked.var(ddof=1) / len(picked) + skipped.var(ddof=1) / len(skipped))
    t = (ml - skipped.mean()) / se if se > 0 else 0.0
    wins = sum(f["ml_better"] for f in folds)
    vs = f"({ml:+.2%} vs {rules:+.2%} per trade for the plain rules)"
    if ml <= 0:
        return False, f"the signals it likes still lose money after fees {vs}"
    if t < 2:
        return False, f"the difference could be luck (t = {t:.1f}, needs 2) {vs}"
    if wins < len(folds) - 1:
        return False, f"it only beat the rules in {wins} of {len(folds)} test periods {vs}"
    return True, f"its picks made money and beat the rules in {wins} of {len(folds)} test periods {vs}"


def push(message, title):
    topic = env("NTFY_TOPIC")
    if not topic:
        print("[no NTFY_TOPIC]", title, "|", message)
        return
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=message.encode(), method="POST",
                                     headers={"Title": title, "Tags": "brain"})
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:
        print("ntfy failed:", e)


def main(histories=None):
    if INTERVAL not in INTERVAL_MIN:
        sys.exit(f"INTERVAL must be one of {', '.join(INTERVAL_MIN)}")
    if histories is None:
        histories = {}
        for coin in COINS:
            h = yahoo_history(coin)
            if h and len(h["c"]) > 300:
                histories[coin] = h
                print(f"{coin}: {len(h['c'])} candles")
            else:
                print(f"{coin}: no usable history, skipped")
    X, y, net, times, coins, hold = build_dataset(histories)
    print(f"{len(X)} historical signals, {y.mean() if len(y) else 0:.0%} profitable after fees")

    DATA.mkdir(exist_ok=True)
    report = {"trained": datetime.now(timezone.utc).isoformat(timespec="seconds"), "interval": INTERVAL,
              "hold_hours": HOLD_HOURS, "fee": FEE, "coins": sorted(histories), "signals": int(len(X)),
              "features": FEATURES, "active": False}

    if len(X) < MIN_SIGNALS:
        report["reason"] = f"only {len(X)} historical signals; needs {MIN_SIGNALS}"
    else:
        folds, oos = walk_forward(X, y, net, times, hold)
        active, reason = judge(folds, oos)
        final = new_model().fit(X, y)
        final.booster_.save_model(str(MODEL_FILE))
        importance = sorted(zip(FEATURES, final.booster_.feature_importance("gain")), key=lambda t: -t[1])
        report.update(active=active, reason=reason, folds=folds, cutoff=float(y.mean()),
                      base_win=float(y.mean()), base_avg=float(net.mean()),
                      top_features=[f for f, _ in importance[:5]])

    REPORT_FILE.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    status = "ON" if report["active"] else "OFF"
    lines = [f"ML is {status}: {report['reason']}.",
             f"Trained on {report['signals']:,} past {INTERVAL} signals from {len(histories)} coins."]
    if report.get("folds"):
        lines.append(f"Plain rules: {report['base_win']:.0%} winners after fees, "
                     f"avg {report['base_avg']:+.2%} per trade over {HOLD_HOURS:g}h.")
        lines.append("Most useful inputs: " + ", ".join(report["top_features"][:3]) + ".")
    push("\n".join(lines), f"ML retrain: {status}")


if __name__ == "__main__":
    main()
