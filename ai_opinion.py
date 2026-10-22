"""AI second opinion for Signal Desk, using TradingAgents.

Modes (set by the 'AI opinion' workflow):
  opinion   - run TradingAgents on each coin the bot just signalled, push the
              verdict to your phone, and log it in data/journal.csv
  scorecard - score past signals 24 hours later and push a summary:
              rules alone vs. when the AI agreed vs. when it disagreed
Both modes first fill in the 24-hour outcome of older journal rows.
"""

import csv
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA = Path("data")
JOURNAL = DATA / "journal.csv"
REPORTS = Path("reports")
FIELDS = ["time_utc", "coin", "interval", "rule_signal", "price", "ai_rating", "ai_view",
          "verdict", "ml_prob", "price_24h", "move_24h_pct", "rule_right"]
KRAKEN_NAMES = {"BTC": "XBT", "DOGE": "XDG"}


def env(name, default=""):
    v = os.environ.get(name, "")
    return v.strip() if v and v.strip() else default


NTFY_TOPIC = env("NTFY_TOPIC")
RUN_URL = (f"{env('GITHUB_SERVER_URL', 'https://github.com')}/{env('GITHUB_REPOSITORY')}"
           f"/actions/runs/{env('GITHUB_RUN_ID')}") if env("GITHUB_RUN_ID") else ""


def push(message, title, tags="", priority="default"):
    if not NTFY_TOPIC:
        print("[no NTFY_TOPIC]", title, "|", message)
        return
    headers = {"Title": title, "Priority": priority}
    if tags:
        headers["Tags"] = tags
    if RUN_URL:
        headers["Click"] = RUN_URL
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}", data=message.encode(),
                                     headers=headers, method="POST")
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:
        print("ntfy failed:", e)


# ---------------------------------------------------------------- journal
def read_journal():
    if not JOURNAL.exists():
        return []
    with JOURNAL.open(newline="") as f:
        return list(csv.DictReader(f))


def write_journal(rows):
    DATA.mkdir(exist_ok=True)
    with JOURNAL.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def kraken_hourly(coin):
    pair = KRAKEN_NAMES.get(coin, coin) + "USD"
    req = urllib.request.Request(f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=60",
                                 headers={"User-Agent": "signal-desk-bot"})
    data = json.loads(urllib.request.urlopen(req, timeout=20).read())
    if data.get("error"):
        raise RuntimeError(", ".join(data["error"]))
    key = next(k for k in data["result"] if k != "last")
    return data["result"][key]


def settle(rows, now=None):
    """Fill in the price 24 hours after each signal and whether the rule was right."""
    now = now or datetime.now(timezone.utc)
    cache = {}
    for r in rows:
        if r.get("move_24h_pct") or not r.get("price"):
            continue
        t = datetime.fromisoformat(r["time_utc"])
        target = t + timedelta(hours=24)
        if now < target + timedelta(hours=1):
            continue
        try:
            hourly = cache.get(r["coin"]) or cache.setdefault(r["coin"], kraken_hourly(r["coin"]))
        except Exception as e:
            print(f"settle {r['coin']}: {e}")
            continue
        later = next((c for c in hourly if int(c[0]) >= target.timestamp()), None)
        if not later:
            continue
        p0, p1 = float(r["price"]), float(later[1])
        move = (p1 / p0 - 1) * 100
        r["price_24h"] = f"{p1:.8g}"
        r["move_24h_pct"] = f"{move:.2f}"
        r["rule_right"] = "yes" if (move > 0) == (r["rule_signal"] == "BUY") else "no"
    return rows


# ---------------------------------------------------------------- TradingAgents
def yahoo_ticker(coin):
    overrides = dict(p.split("=", 1) for p in env("YAHOO_TICKERS").split(",") if "=" in p)
    coin = {"XBT": "BTC", "XDG": "DOGE"}.get(coin, coin)
    return overrides.get(coin, f"{coin}-USD").strip()


def view_of(rating):
    r = (rating or "").lower()
    if r in ("buy", "overweight"):
        return "bullish"
    if r in ("sell", "underweight"):
        return "bearish"
    if r == "hold":
        return "neutral"
    return "unclear"


def verdict_of(rule_signal, view):
    if view in ("neutral", "unclear"):
        return "unsure"
    return "agrees" if (view == "bullish") == (rule_signal == "BUY") else "disagrees"


def short_reason(text, limit=320):
    text = re.sub(r"[#*_>`|]+", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


# Cost-conscious defaults per provider: (key secret, quick model, deep model)
PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "claude-haiku-4-5", "claude-sonnet-5"),
    "openai": ("OPENAI_API_KEY", "gpt-6-luna", "gpt-6-sol"),
    "google": ("GOOGLE_API_KEY", "gemini-3.8-flash", "gemini-3.8-flash"),
    "deepseek": ("DEEPSEEK_API_KEY", "deepseek-flash", "deepseek-v4-pro"),
}


def pick_llm():
    """Use AI_PROVIDER if set, otherwise the first provider whose key secret is present."""
    provider = env("AI_PROVIDER").lower()
    if not provider:
        provider = next((p for p, (key, _, _) in PROVIDERS.items() if env(key)), "")
    if not provider:
        raise RuntimeError("No LLM API key found. Add ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                           "GOOGLE_API_KEY or DEEPSEEK_API_KEY as a repository secret.")
    _, quick, deep = PROVIDERS.get(provider, ("", "", ""))
    return provider, env("AI_QUICK_MODEL", quick), env("AI_DEEP_MODEL", deep)


def make_graph():
    from tradingagents.default_config import build_default_config
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = build_default_config()
    provider, quick, deep = pick_llm()
    if not (quick and deep):
        raise RuntimeError(f"Set AI_QUICK_MODEL and AI_DEEP_MODEL for provider '{provider}'.")
    config.update(llm_provider=provider, quick_think_llm=quick, deep_think_llm=deep)
    config["memory_log_path"] = str((DATA / "trading_memory.md").resolve())
    config["holding_period_days"] = 1
    analysts = [a.strip() for a in env("AI_ANALYSTS", "market,social,news").split(",") if a.strip()]
    print("TradingAgents:", config["llm_provider"], config["quick_think_llm"], "/",
          config["deep_think_llm"], "analysts", analysts)
    return TradingAgentsGraph(selected_analysts=analysts, config=config)


def opinion(items):
    rows = settle(read_journal())
    graph = make_graph()
    REPORTS.mkdir(exist_ok=True)
    for it in items:
        coin, sig = it["coin"], it["signal"]
        ticker = yahoo_ticker(coin)
        row = {"time_utc": it["time"], "coin": coin, "interval": it.get("interval", ""),
               "rule_signal": sig, "price": f"{float(it['price']):.8g}",
               "ml_prob": "" if it.get("ml_prob") is None else f"{float(it['ml_prob']):.3f}"}
        try:
            date = datetime.fromisoformat(it["time"]).astimezone(timezone.utc).strftime("%Y-%m-%d")
            t0 = time.time()
            state, rating = graph.propagate(ticker, date, asset_type="crypto")
            print(f"{ticker}: {rating} in {time.time() - t0:.0f}s")
            view = view_of(rating)
            verdict = verdict_of(sig, view)
            row.update(ai_rating=str(rating), ai_view=view, verdict=verdict)
            try:
                folder = REPORTS / f"{date}-{coin}"
                graph.save_reports(state, ticker, save_path=folder)
            except Exception as e:
                print("report save failed:", e)
            icon = {"agrees": "white_check_mark", "disagrees": "warning", "unsure": "grey_question"}[verdict]
            head = {"agrees": f"AI agrees with {sig} {coin}",
                    "disagrees": f"AI disagrees with {sig} {coin}",
                    "unsure": f"AI unsure about {sig} {coin}"}[verdict]
            push(f"Rating: {rating} ({view})\n{short_reason(state.get('final_trade_decision', ''))}\n"
                 f"Tap for the full report.", title=head, tags=icon,
                 priority="high" if verdict == "disagrees" else "default")
        except Exception as e:
            print(f"{coin}: AI check failed: {e!r}")
            row.update(ai_rating="ERROR", ai_view="", verdict="error")
            push(f"The AI check for {sig} {coin} failed: {str(e)[:200]}\nThe rule signal still stands.",
                 title=f"AI check failed: {coin}", tags="x")
        rows.append(row)
        write_journal(rows)  # save after each coin, in case a later one crashes the run


# ---------------------------------------------------------------- scorecard
def ml_cutoff():
    try:
        return float(json.loads((DATA / "ml_report.json").read_text())["cutoff"])
    except Exception:
        return None


def scorecard_text(rows):
    done = [r for r in rows if r.get("move_24h_pct") and r.get("verdict") not in ("", "error")]
    if not done:
        return "No signals have been scored yet. Each one is scored 24 hours after it fires."

    def line(label, group):
        if not group:
            return f"{label}: none yet"
        signed = [float(r["move_24h_pct"]) * (1 if r["rule_signal"] == "BUY" else -1) for r in group]
        hits = sum(1 for r in group if r["rule_right"] == "yes")
        return (f"{label}: {len(group)} signals, right {hits / len(group):.0%}, "
                f"avg {sum(signed) / len(signed):+.2f}% per trade")

    out = [line("All rule signals", done),
           line("AI agreed", [r for r in done if r["verdict"] == "agrees"]),
           line("AI disagreed", [r for r in done if r["verdict"] == "disagrees"]),
           line("AI unsure", [r for r in done if r["verdict"] == "unsure"])]
    cutoff = ml_cutoff()
    scored = [r for r in done if r.get("ml_prob")]
    if scored and cutoff is not None:
        out += [line("ML liked", [r for r in scored if float(r["ml_prob"]) > cutoff]),
                line("ML weak", [r for r in scored if float(r["ml_prob"]) <= cutoff])]
    out.append("Moves are 24h after the signal, before Kraken's ~0.8% round-trip fee. "
               "Trust a filter only if its line clearly beats 'All' over 30+ signals.")
    return "\n".join(out)


def scorecard():
    rows = settle(read_journal())
    write_journal(rows)
    text = scorecard_text(rows)
    print(text)
    push(text, title="Weekly scorecard: rules vs AI", tags="bar_chart")


def main():
    mode = env("MODE", "opinion")
    if mode == "scorecard":
        scorecard()
        return
    items = json.loads(env("ITEMS", "[]"))
    if not items:
        sys.exit("No signals given (ITEMS is empty).")
    opinion(items)


if __name__ == "__main__":
    main()
