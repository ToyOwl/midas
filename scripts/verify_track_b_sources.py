"""Живая проверка источников трека B: USDRUB (ISS), BZ=F/^TNX, earnings, дропы."""
from pathlib import Path

from portagent.collect.dividends import fetch_us_dividends
from portagent.collect.earnings import fetch_us_earnings
from portagent.collect.macro import fetch_usdrub, fetch_yf_series

print("== USDRUB (ISS SELT, USD000UTSTOM) ==")
try:
    frame = fetch_usdrub(from_date="2021-11-01", till_date="2026-11-12")
    print(f"строк: {len(frame)}; {frame.index[0]}..{frame.index[-1]}; "
          f"последний курс {frame['value'].iloc[-1]:.2f}")
except Exception as error:
    print("ОШИБКА:", str(error)[:160])

for symbol in ("BZ=F", "^TNX"):
    print(f"== {symbol} (yfinance) ==")
    try:
        frame = fetch_yf_series(symbol)
        print(f"строк: {len(frame)}; последний {frame['value'].iloc[-1]:.2f}")
    except Exception as error:
        print("ОШИБКА:", str(error)[:160])

print("== earnings AAPL ==")
try:
    dates = fetch_us_earnings("AAPL")
    print(f"дат: {len(dates)}; ближ.: {dates[:3]}")
except Exception as error:
    print("ОШИБКА:", str(error)[:160])

print("== дивиденды AAPL (справочно) ==")
try:
    rows = fetch_us_dividends("AAPL")
    print(f"выплат: {len(rows)}; последние: {rows[-2:]}")
except Exception as error:
    print("ОШИБКА:", str(error)[:160])
