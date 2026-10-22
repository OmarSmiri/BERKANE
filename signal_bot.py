"""Signal Desk bot + AI second opinion.

Fast layer (this file): checks Kraken every minute on GitHub's servers, computes
BUY / SELL / HOLD from EMA20/50 trend, MACD and RSI on closed candles only, and
sends ntfy alerts to your phone when a coin switches, during your alert hours.

Slow layer: when a coin switches to BUY or SELL and an LLM API key is set, it
asks the "AI opinion" workflow (ai_opinion.py, running TradingAgents) to check
the same coin. That answer arrives on your phone a few minutes later.
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from features import Indicators, features_at

TZ = ZoneInfo("Europe/Oslo")
STATE_FILE = Path("state/signals-v2.json")


def env(name, default=""):
    v = os.environ.get(name, "")
    return v.strip() if v and v.strip() else default


COINS = [c.strip().upper() for c in env("COINS", "BTC,ETH,SOL").split(",") if c.strip()]
INTERVAL = env("INTERVAL", "15m")
DEFAULT_HOURS = "Mon-Fri 14:00-18:00; Daily 01:30-04:30"
ALWAYS_ON = env("ALWAYS_ON", "").lower() in ("1", "true", "yes", "on")
FOMC_DATES = [d.strip() for d in env("FOMC_DATES", "2026-10-28,2026-12-09").split(",") if d.strip()]
NTFY_TOPIC = env("NTFY_TOPIC")
RUN_MINUTES = int(env("RUN_MINUTES", "350"))

AI_KEY_SET = env("AI_KEY_SET", "false").lower() == "true"
AI_ENABLED = AI_KEY_SET and env("AI_ENABLED", "true").lower() not in ("0", "false", "no", "off")
AI_MAX_PER_DAY = int(env("AI_MAX_PER_DAY", "6"))
ML_HIDE_WEAK = env("ML_HIDE_WEAK", "false").lower() in ("1", "true", "yes", "on")

INTERVAL_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
KRAKEN_NAMES = {"BTC": "XBT", "DOGE": "XDG"}
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


# ---------------------------------------------------------------- alert hours
def parse_days(spec):
    s = spec.strip().lower()
    if s in ("daily", "every", "all"):
        return set(range(7))
    if s == "weekdays":
        return set(range(5))
    if s == "weekends":
        return {5, 6}
    out = set()
    for part in s.split(","):
        part = part.strip()[:3] if "-" not in part else part.strip()
        if "-" in part:
            a, b = [p.strip()[:3] for p in part.split("-")]
            i, j = DAYS.index(a), DAYS.index(b)
            k = i
            while True:
                out.add(k)
                if k == j:
                    break
                k = (k + 1) % 7
        elif part:
            out.add(DAYS.index(part))
    return out


def hm(text):
    h, m = text.strip().split(":")
    return int(h) * 60 + int(m)


def parse_hours(text):
    """'Mon-Fri 14:00-18:00; Daily 01:30-04:30' -> [(days, start, end, label)]."""
    windows = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        days, times = chunk.rsplit(" ", 1)
        a, b = times.split("-")
        windows.append((parse_days(days), hm(a), hm(b), chunk))
    return windows


WINDOWS = parse_hours(env("ALERT_HOURS", DEFAULT_HOURS))


def active_window(now):
    """Return the label of the alert window that contains `now` (Oslo time), or None."""
    if ALWAYS_ON:
        return "all day"
    mins = now.hour * 60 + now.minute
    wd = now.weekday()
    for days, start, end, label in WINDOWS:
        if start <= end:
            if wd in days and start <= mins < end:
                return label
        else:  # runs past midnight: the window belongs to the day it starts on
            if wd in days and mins >= start:
                return label
            if (wd - 1) % 7 in days and mins < end:
                return label
    if now.strftime("%Y-%m-%d") in FOMC_DATES and hm("19:30") <= mins < hm("21:30"):
        return "Fed decision evening"
    return None


# ---------------------------------------------------------------- indicators + ML
def signal(closes):
    """Rule signal on the latest closed candle (same code the ML trainer replays)."""
    ind = Indicators(closes)
    i = len(closes) - 1
    info = {"ema20": ind.e20[i], "ema50": ind.e50[i], "macd_hist": ind.hist[i], "rsi": ind.rsi[i]}
    return ind.signal_at(i), info, ind


ML = None  # (booster, cutoff) when a trained model is switched on


def load_ml():
    """Load data/ml_model.txt if the weekly retrain switched it on for this candle size."""
    global ML
    ML = None
    if env("ML_ENABLED", "true").lower() in ("0", "false", "no", "off"):
        return "off (ML_ENABLED=false)"
    try:
        report = json.loads(Path("data/ml_report.json").read_text())
    except Exception:
        return "not trained yet"
    if report.get("interval") != INTERVAL:
        return f"off (trained on {report.get('interval')} candles, you use {INTERVAL}; retrain)"
    if not report.get("active"):
        return f"off ({report.get('reason', 'did not pass its test')})"
    try:
        import lightgbm as lgb
        ML = (lgb.Booster(model_file="data/ml_model.txt"), float(report["cutoff"]))
        return "on"
    except Exception as e:
        return f"off (could not load model: {e})"


def ml_score(ind, rows, sig):
    """Model's estimated chance this signal is profitable after fees, or None."""
    if not ML:
        return None
    try:
        i = len(rows) - 1
        x = features_at(ind, i, 1 if sig == "BUY" else -1, [int(r[0]) for r in rows],
                        [float(r[2]) for r in rows], [float(r[3]) for r in rows],
                        [float(r[6]) for r in rows])
        return float(ML[0].predict([x])[0])
    except Exception as e:
        print("ML score failed:", e)
        return None


# ---------------------------------------------------------------- I/O
def http_json(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=20) as r:
        body = r.read()
        return json.loads(body) if body else {}


def candles(coin):
    """Closed Kraken candles for COIN/USD: rows of [time, o, h, l, c, vwap, vol, count]."""
    pair = KRAKEN_NAMES.get(coin, coin) + "USD"
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={INTERVAL_MIN[INTERVAL]}"
    data = http_json(url, headers={"User-Agent": "signal-desk-bot"})
    if data.get("error"):
        raise RuntimeError(", ".join(data["error"]))
    key = next(k for k in data["result"] if k != "last")
    return data["result"][key][:-1]  # last row is the candle still forming


def push(message, title="Signal Desk", tags="", priority="default", click=""):
    if not NTFY_TOPIC:
        print("[no NTFY_TOPIC]", title, "|", message)
        return
    headers = {"Title": title, "Priority": priority}
    if tags:
        headers["Tags"] = tags
    if click:
        headers["Click"] = click
    try:
        http_json(f"https://ntfy.sh/{NTFY_TOPIC}", data=message.encode(), headers=headers, method="POST")
    except Exception as e:
        print("ntfy failed:", e)


def request_ai_opinion(items):
    """Start the 'AI opinion' workflow for these signals. Returns True if GitHub accepted it."""
    repo, token = env("GITHUB_REPOSITORY"), env("GITHUB_TOKEN")
    if not (repo and token):
        print("AI: no GitHub token, skipping")
        return False
    url = f"https://api.github.com/repos/{repo}/actions/workflows/ai-opinion.yml/dispatches"
    body = json.dumps({"ref": env("GITHUB_REF_NAME", "main"),
                       "inputs": {"mode": "opinion", "items": json.dumps(items)}}).encode()
    try:
        http_json(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "Content-Type": "application/json", "User-Agent": "signal-desk-bot"})
        return True
    except Exception as e:
        print("AI dispatch failed:", e)
        return False


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=1))


def fmt_price(p):
    return f"${p:,.2f}" if p >= 1 else f"${p:.6f}"


# ---------------------------------------------------------------- main check
def check(state, now=None):
    now = now or datetime.now(TZ)
    window = active_window(now)
    coins = state.setdefault("coins", {})
    ai_queue = []

    for coin in COINS:
        try:
            rows = candles(coin)
        except Exception as e:
            print(f"{coin}: fetch failed ({e})")
            continue
        if not rows:
            continue
        closes = [float(r[4]) for r in rows]
        last_t = int(rows[-1][0])
        cur = coins.setdefault(coin, {"signal": None, "since": None, "alerted": True, "candle": 0})
        if last_t == cur["candle"]:
            continue  # no new closed candle since last look
        cur["candle"] = last_t
        sig, info, ind = signal(closes)
        price = closes[-1]

        if sig != cur["signal"]:
            first_look = cur["signal"] is None
            ml = ml_score(ind, rows, sig) if sig in ("BUY", "SELL") else None
            cur.update(signal=sig, since=now.isoformat(), price=price, ml=ml, info=info)
            # First look after a restart: remember the signal, don't alert on it.
            cur["alerted"] = first_look or sig == "HOLD"
            print(f"{coin}: -> {sig} at {fmt_price(price)} {info}")

        if not cur["alerted"] and sig in ("BUY", "SELL") and window:
            since = datetime.fromisoformat(cur["since"])
            late = "" if now - since < timedelta(minutes=INTERVAL_MIN[INTERVAL] + 2) else \
                f" (switched {since.strftime('%a %H:%M')})"
            cur["alerted"] = True
            ml = cur.get("ml")
            ml_note, weak = "", False
            if ml is not None:
                weak = ml <= ML[1]
                ml_note = (f"\nML confidence {ml:.0%} — "
                           + ("weak, below-average odds" if weak else "above-average odds"))
            if weak and ML_HIDE_WEAK:
                print(f"{coin}: {sig} hidden, ML confidence {ml:.0%}")
                continue
            ai_note = ""
            if AI_ENABLED and ai_budget(state, now) - len(ai_queue) > 0:
                ai_queue.append({"coin": coin, "signal": sig, "price": price,
                                 "time": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
                                 "interval": INTERVAL, "ml_prob": ml})
                ai_note = "\nAI second opinion coming in a few minutes."
            push(f"{sig} {coin} at {fmt_price(price)}{late}\n"
                 f"RSI {info.get('rsi', 0):.0f} · {INTERVAL} candles{ml_note}{ai_note}",
                 title=f"{sig} {coin}" + (" (weak)" if weak else ""),
                 tags="chart_with_upwards_trend" if sig == "BUY" else "chart_with_downwards_trend",
                 priority="default" if weak else "high")

    if ai_queue and request_ai_opinion(ai_queue):
        day = now.strftime("%Y-%m-%d")
        used = state.setdefault("ai_used", {})
        used[day] = used.get(day, 0) + len(ai_queue)
        for k in list(used):
            if k != day:
                del used[k]
    save_state(state)


def ai_budget(state, now):
    return AI_MAX_PER_DAY - state.get("ai_used", {}).get(now.strftime("%Y-%m-%d"), 0)


def main():
    if INTERVAL not in INTERVAL_MIN:
        sys.exit(f"INTERVAL must be one of {', '.join(INTERVAL_MIN)}")
    state = load_state()
    hours = "all day" if ALWAYS_ON else env("ALERT_HOURS", DEFAULT_HOURS)
    ai = (f"on, up to {AI_MAX_PER_DAY} checks a day" if AI_ENABLED else
          "off (add an LLM API key secret to turn it on)" if not AI_KEY_SET else "off (AI_ENABLED=false)")
    ml = load_ml()
    print(f"Coins {COINS}, {INTERVAL} candles, alert hours: {hours}, AI: {ai}, ML: {ml}")
    if not state.get("started"):
        push(f"Signal Desk bot is running.\nCoins: {', '.join(COINS)} · {INTERVAL} candles\n"
             f"Alert hours (Oslo time): {hours}, plus Fed decision evenings.\n"
             f"AI second opinion: {ai}.\nML confidence: {ml}.")
        state["started"] = True
    elif state.get("ml_status") and state["ml_status"] != ml and (ml == "on" or state["ml_status"] == "on"):
        push(f"ML confidence is now {ml}.", title="Signal Desk ML")
    state["ml_status"] = ml
    end = time.time() + RUN_MINUTES * 60
    while time.time() < end:
        t0 = time.time()
        try:
            check(state)
        except Exception as e:
            print("check failed:", e)
        time.sleep(max(1, 60 - (time.time() - t0)))


if __name__ == "__main__":
    main()
