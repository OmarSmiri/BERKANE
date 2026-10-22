"""Indicators, rule signals and ML features, shared by the bot and the trainer.

Pure Python (no numpy) so the bot can use it without extra installs. Every
value at candle i uses only candles 0..i, so nothing peeks into the future.
"""

import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Oslo")

FEATURES = [
    "side", "rsi", "macd_hist", "ema_gap", "ema20_slope",
    "ret_1", "ret_4", "ret_16", "ret_96",
    "volatility", "range", "volume_ratio", "hour", "weekday",
]


def ema_series(values, n):
    k = 2 / (n + 1)
    out, e = [], values[0]
    for v in values:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi_series(values, n=14):
    out = [50.0] * len(values)
    if len(values) <= n:
        return out
    gains = losses = 0.0
    for i in range(1, n + 1):
        d = values[i] - values[i - 1]
        gains += max(d, 0)
        losses += max(-d, 0)
    ag, al = gains / n, losses / n
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n + 1, len(values)):
        d = values[i] - values[i - 1]
        ag = (ag * (n - 1) + max(d, 0)) / n
        al = (al * (n - 1) + max(-d, 0)) / n
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


class Indicators:
    """All indicator series for one price history."""

    def __init__(self, closes):
        self.c = closes
        self.e20 = ema_series(closes, 20)
        self.e50 = ema_series(closes, 50)
        macd = [a - b for a, b in zip(ema_series(closes, 12), ema_series(closes, 26))]
        sig = ema_series(macd, 9)
        self.hist = [m - s for m, s in zip(macd, sig)]
        self.rsi = rsi_series(closes)

    def signal_at(self, i):
        """BUY when trend, momentum and RSI agree upward; SELL when they agree downward."""
        if i < 59:
            return "HOLD"
        e20, e50, h, r = self.e20[i], self.e50[i], self.hist[i], self.rsi[i]
        if e20 > e50 and h > 0 and r < 70:
            return "BUY"
        if e20 < e50 and h < 0 and r > 30:
            return "SELL"
        return "HOLD"


def features_at(ind, i, side, times, highs, lows, volumes):
    """Feature vector for a signal at candle i. side is +1 for BUY, -1 for SELL.

    Direction-sensitive values are multiplied by side, so for both BUY and SELL
    a positive number means "in favour of the signal".
    """
    c = ind.c
    p = c[i]

    def ret(n):
        j = max(0, i - n)
        return side * (p / c[j] - 1)

    rets = [c[j] / c[j - 1] - 1 for j in range(max(1, i - 47), i + 1)]
    mean = sum(rets) / len(rets)
    vol = math.sqrt(sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1))
    rng = sum((highs[j] - lows[j]) / c[j] for j in range(max(0, i - 13), i + 1)) / min(14, i + 1)
    past_v = volumes[max(0, i - 48):i]
    avg_v = sum(past_v) / len(past_v) if past_v else 0
    vol_ratio = volumes[i] / avg_v if avg_v > 0 else 1.0
    t = datetime.fromtimestamp(times[i], timezone.utc).astimezone(TZ)
    j5 = max(0, i - 5)
    return [
        side,
        ind.rsi[i] if side > 0 else 100 - ind.rsi[i],
        side * ind.hist[i] / p,
        side * (ind.e20[i] - ind.e50[i]) / p,
        side * (ind.e20[i] - ind.e20[j5]) / p,
        ret(1), ret(4), ret(16), ret(96),
        vol, rng, vol_ratio, t.hour, t.weekday(),
    ]


def signal_events(ind):
    """Indices where the signal switches to BUY or SELL (what the bot alerts on)."""
    events, prev = [], None
    for i in range(len(ind.c)):
        s = ind.signal_at(i)
        if prev is not None and s != prev and s in ("BUY", "SELL"):
            events.append((i, s))
        prev = s
    return events
