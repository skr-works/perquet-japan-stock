# scripts/build_latest_parquet.py
from __future__ import annotations

import os
import sys
import time
import json
import math
import random
import shutil
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# =========================
# 設定
# =========================

MASTER_CSV_URL_DEFAULT = "https://raw.githubusercontent.com/skr-works/ticker-master-japan/refs/heads/main/data/master.csv"
MASTER_CSV_URL = os.getenv("MASTER_CSV_URL", MASTER_CSV_URL_DEFAULT)

# 出力ファイル名（Releasesに載せる）
OUT_NAME = os.getenv("OUT_NAME", "latest_stock_data.parquet")
TMP_NAME = os.getenv("TMP_NAME", "temp_stock_data.parquet")

# 実行制御（IPブロック対策）
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "50"))
CHUNK_SLEEP_MIN = float(os.getenv("CHUNK_SLEEP_MIN", "3.0"))
CHUNK_SLEEP_MAX = float(os.getenv("CHUNK_SLEEP_MAX", "5.0"))
PER_TICKER_SLEEP_MIN = float(os.getenv("PER_TICKER_SLEEP_MIN", "0.03"))
PER_TICKER_SLEEP_MAX = float(os.getenv("PER_TICKER_SLEEP_MAX", "0.15"))

# 失敗の全体傾向で止める（誤検知を避ける）
# 直近 WINDOW 件に対して、throttle系が THROTTLE_RATIO を超えたら終了
THROTTLE_WINDOW = int(os.getenv("THROTTLE_WINDOW", "80"))
THROTTLE_RATIO = float(os.getenv("THROTTLE_RATIO", "0.45"))

# バリデーション（最低限）
MIN_ROWS = int(os.getenv("MIN_ROWS", "3500"))
# Parquet(zstd)は圧縮が強く、サイズで品質判定すると偽陽性が起きる。
# デフォルトではサイズ検査を無効化（MIN_BYTES <= 0 でスキップ）。
MIN_BYTES = int(os.getenv("MIN_BYTES", "0"))
MAX_PRICE_NA_RATIO = float(os.getenv("MAX_PRICE_NA_RATIO", "0.95"))


# =========================
# スキーマ（厳格）
# ※「code」を追加して30列にしています（参照側が銘柄コードで引けるようにするため）
#   どうしても29列固定なら、最後に df.drop(columns=["code"]) してください。
# =========================
SCHEMA = pa.schema([
    ("code", pa.string()),
    ("name", pa.string()),
    ("sector", pa.string()),
    ("price", pa.float64()),
    ("target_price", pa.float64()),
    ("upside", pa.float64()),
    ("grade", pa.string()),
    ("peg", pa.float64()),
    ("roic", pa.float64()),
    ("final_g", pa.float64()),
    ("psr", pa.float64()),
    ("equity_ratio", pa.float64()),
    ("div_yield", pa.float64()),
    ("div_streak", pa.int32()),
    ("market_cap", pa.float64()),   # 億
    ("roe", pa.float64()),
    ("roa", pa.float64()),
    ("op_cf_margin", pa.float64()),
    ("per", pa.float64()),
    ("pbr", pa.float64()),
    ("adj_pbr", pa.float64()),
    ("cash", pa.float64()),         # 億
    ("debt", pa.float64()),         # 億
    ("net_de_ratio", pa.float64()),
    ("g_raw", pa.float64()),
    ("op_raw", pa.float64()),
    ("earnings_alert", pa.string()),
    ("earnings_date", pa.string()),  # YYYY-MM-DD or null
    ("latest_i_int", pa.float64()),  # 億
    ("k_int", pa.float64()),         # 億
])


# =========================
# ユーティリティ
# =========================

TO_OKU = 100_000_000.0


def _to_float(x: Any) -> float:
    try:
        if x is None:
            return np.nan
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        v = pd.to_numeric(x, errors="coerce")
        return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan


def _to_int32_nullable(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return None
        return int(v)
    except Exception:
        return None


def _safe_div(a: float, b: float) -> float:
    if b is None or b == 0 or pd.isna(b):
        return np.nan
    if a is None or pd.isna(a):
        return np.nan
    return a / b


def _round_or_nan(x: float, ndigits: int) -> float:
    if x is None or pd.isna(x):
        return np.nan
    try:
        return round(float(x), ndigits)
    except Exception:
        return np.nan


def _is_throttle_error(msg: str) -> bool:
    m = (msg or "").lower()
    keys = ["429", "too many requests", "rate limit", "forbidden", "403", "temporarily unavailable"]
    return any(k in m for k in keys)


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _sleep_rand(a: float, b: float) -> None:
    time.sleep(random.uniform(a, b))


# =========================
# 指標計算（yfinance）
# =========================

def _get_value(df: pd.DataFrame, keys: List[str], date_col: Any) -> float:
    if df is None or df.empty or date_col is None or date_col not in df.columns:
        return np.nan
    for k in keys:
        if k in df.index:
            return _to_float(df.loc[k, date_col])
    return np.nan


def _get_eps_for_date(financials: pd.DataFrame, date_col: Any) -> float:
    # Basic EPSがあればそれ、なければ NetIncome / Shares を試みる
    eps = _get_value(financials, ["Basic EPS", "Earnings Per Share Basic"], date_col)
    if not pd.isna(eps) and eps != 0:
        return eps

    ni = _get_value(financials, ["Net Income", "Net Income Common Stockholders"], date_col)
    shares = _get_value(financials, ["Basic Average Shares", "Average Diluted Shares"], date_col)
    if (not pd.isna(ni)) and (not pd.isna(shares)) and shares > 0:
        return ni / shares
    return np.nan


def _calc_growth_median(financials: pd.DataFrame, dates: List[Any]) -> Tuple[float, float, List[float]]:
    """
    EPS成長率の簡易推計：
    - eps_old <= 0 は除外
    - g > 500% は除外
    """
    g_list: List[float] = []
    if financials is None or financials.empty or len(dates) < 2:
        return np.nan, np.nan, g_list

    for i in range(min(len(dates) - 1, 5)):
        d_new = dates[i]
        d_old = dates[i + 1]
        eps_new = _get_eps_for_date(financials, d_new)
        eps_old = _get_eps_for_date(financials, d_old)
        if pd.isna(eps_new) or pd.isna(eps_old) or eps_old <= 0:
            continue
        g = ((eps_new - eps_old) / abs(eps_old)) * 100.0
        if g > 500.0:
            continue
        g_list.append(float(g))

    if len(g_list) >= 3:
        final_g = float(np.median(g_list))
    elif len(g_list) >= 1:
        final_g = float(np.mean(g_list))
    else:
        final_g = np.nan

    # キャップ（あなたの既存ロジックに合わせる）
    if not pd.isna(final_g) and final_g > 30.0:
        final_g = 30.0

    g_raw = g_list[0] if g_list else np.nan
    return final_g, g_raw, g_list


def _calc_dividend_metrics(stock: yf.Ticker, current_price: float) -> Tuple[float, Optional[int]]:
    """
    div_yield(%), div_streak(年) を推計
    """
    try:
        dividends = stock.dividends
    except Exception:
        return np.nan, None

    if dividends is None or getattr(dividends, "empty", True) or pd.isna(current_price) or current_price <= 0:
        return np.nan, None

    div_yield = np.nan
    streak = 0
    try:
        tz = dividends.index.tz
        one_year_ago = pd.Timestamp.now(tz=tz) - pd.DateOffset(years=1)
        recent = dividends[dividends.index >= one_year_ago]
        annual_amt = float(recent.sum()) if recent is not None else 0.0
        div_yield = (annual_amt / float(current_price)) * 100.0
        if div_yield > 50.0:
            div_yield = np.nan
    except Exception:
        pass

    try:
        s = dividends.copy()
        s.index = s.index.tz_localize(None)
        yearly = s.resample("YE").sum().sort_index(ascending=False)
        if yearly.empty or len(yearly) < 2:
            return div_yield, 0

        current_year = dt.datetime.now().year
        start = 0
        latest_year = yearly.index[0].year
        if latest_year >= current_year:
            start = 1

        streak = 0
        for i in range(start, len(yearly) - 1):
            if yearly.iloc[i] > yearly.iloc[i + 1]:
                streak += 1
            elif yearly.iloc[i] == yearly.iloc[i + 1]:
                break
            else:
                break
    except Exception:
        return div_yield, None

    return div_yield, int(streak)


def _calc_earnings(stock: yf.Ticker) -> Tuple[Optional[str], Optional[str]]:
    """
    earnings_date(YYYY-MM-DD), earnings_alert("1ヶ月以内" or null)
    """
    earnings_date = None
    earnings_alert = None
    try:
        cal = stock.calendar
        dates = []
        if isinstance(cal, dict):
            if "Earnings Date" in cal:
                dates = cal["Earnings Date"]
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            # ざっくり拾う
            for col in cal.columns:
                if "Earnings" in str(col):
                    dates = cal[col].tolist()
                    break
            if not dates:
                dates = cal.iloc[:, 0].tolist()

        if dates:
            d0 = dates[0]
            if isinstance(d0, (dt.datetime, dt.date)):
                d0_date = d0.date() if isinstance(d0, dt.datetime) else d0
                earnings_date = d0_date.strftime("%Y-%m-%d")
                today = dt.datetime.now().date()
                days = (d0_date - today).days
                if 0 <= days <= 30:
                    earnings_alert = "1ヶ月以内"
    except Exception:
        pass
    return earnings_alert, earnings_date


def _calc_intangibles_adj_pbr(
    financials: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    equity: float,
    market_cap_yen: float
) -> Tuple[float, float, float]:
    """
    latest_i_int(円), k_int(円), adj_pbr を推計（あなたの既存ロジック簡易移植）
    """
    if financials is None or financials.empty or balance_sheet is None or balance_sheet.empty:
        return np.nan, np.nan, np.nan

    years = list(financials.columns)

    def _get_series(keys: List[str]) -> pd.Series:
        for k in keys:
            if k in financials.index:
                s = financials.loc[k].fillna(0)
                try:
                    s = s.astype("float64").abs()
                except Exception:
                    s = s.apply(lambda v: abs(pd.to_numeric(v, errors="coerce") or 0.0))
                return s
        return pd.Series(0.0, index=years)

    try:
        rnd = _get_series(["Research And Development", "Research Development", "Research & Development"])
        sga = _get_series([
            "Selling General And Administration",
            "Selling General and Administrative",
            "Selling, General And Administration",
            "Selling, General and Administrative"
        ])
        i_int = rnd + (0.3 * sga)

        k_int = 0.0
        delta = 0.2
        for i, val in enumerate(i_int):
            if pd.notna(val):
                k_int += abs(float(val)) * ((1.0 - delta) ** i)

        latest_i_int = abs(float(i_int.iloc[0])) if not i_int.empty else np.nan

        latest_date_bs = balance_sheet.columns[0] if not balance_sheet.empty else None
        total_intangibles = _get_value(
            balance_sheet,
            ["Goodwill And Other Intangible Assets", "Intangible Assets", "Other Intangible Assets"],
            latest_date_bs
        )
        if pd.isna(total_intangibles) or total_intangibles == 0:
            goodwill = _get_value(balance_sheet, ["Goodwill"], latest_date_bs)
            other = _get_value(balance_sheet, ["Other Intangible Assets"], latest_date_bs)
            total_intangibles = (0 if pd.isna(goodwill) else goodwill) + (0 if pd.isna(other) else other)

        tangible = equity - (0 if pd.isna(total_intangibles) else total_intangibles)
        adjusted_equity = tangible + k_int

        if (not pd.isna(market_cap_yen)) and market_cap_yen > 0 and adjusted_equity > 0:
            adj_pbr = market_cap_yen / adjusted_equity
        else:
            adj_pbr = np.nan

        return latest_i_int, k_int, adj_pbr
    except Exception:
        return np.nan, np.nan, np.nan


def _grade_and_target(
    current_price: float,
    per: float,
    roic: float,
    final_g: float,
    psr: float,
    equity_ratio: float,
    net_de_ratio: float,
    div_yield: float,
    div_streak: Optional[int],
    op_cf_3yr_avg: float,
    adj_pbr: float,
    pbr: float
) -> Tuple[Optional[str], float, float]:
    """
    grade, target_price, upside を推計（あなたの既存ロジック準拠）
    """
    valid_g = 0.0 if pd.isna(final_g) else float(final_g)

    score = 0
    is_value = False

    # ROIC
    if not pd.isna(roic):
        if roic >= 15:
            score += 3
        elif roic >= 10:
            score += 2
        elif roic >= 8:
            score += 1

    # 成長率
    if valid_g >= 15:
        score += 3
    elif valid_g >= 7:
        score += 2
    elif valid_g > 0:
        score += 1

    # 割安
    if (not pd.isna(psr)) and psr > 0 and psr < 0.8:
        score += 1
        is_value = True

    if (not pd.isna(adj_pbr)) and adj_pbr > 0:
        if (not pd.isna(pbr)) and pbr > 0 and adj_pbr <= 1.5 and ((pbr - adj_pbr) / pbr) >= 0.3:
            score += 2
            is_value = True
        elif adj_pbr < 1.0:
            score += 1
            is_value = True

    # 財務
    if not pd.isna(net_de_ratio):
        if net_de_ratio < 0:
            score += 1
        elif net_de_ratio > 1.0:
            score -= 1

    if (not pd.isna(div_yield)) and div_yield >= 3.0:
        score += 1

    if div_streak is not None and div_streak >= 3:
        score += 1

    if (not pd.isna(equity_ratio)) and equity_ratio < 20:
        score -= 1

    if (not pd.isna(op_cf_3yr_avg)) and op_cf_3yr_avg < 0:
        score -= 3

    grade = "D"
    if score >= 7 and (not pd.isna(op_cf_3yr_avg)) and op_cf_3yr_avg > 0 and equity_ratio >= 30:
        if score >= 9 and (not pd.isna(roic)) and roic >= 10 and valid_g >= 10 and is_value:
            grade = "AA"
        else:
            grade = "A"
    elif score >= 4 and (not pd.isna(op_cf_3yr_avg)) and op_cf_3yr_avg > 0:
        grade = "B"
    elif score >= 2:
        grade = "C"

    # target_per
    target_per = 15.0
    if valid_g > 20:
        target_per = 25.0
    elif valid_g > 10:
        target_per = 20.0
    elif valid_g < 0:
        target_per = 10.0

    if not pd.isna(roic):
        if roic > 15:
            target_per *= 1.15
        elif roic >= 10:
            target_per *= 1.05
        elif roic < 5:
            target_per *= 0.9

    if target_per > 25:
        target_per = 25.0

    if pd.isna(current_price) or current_price <= 0:
        return grade, np.nan, np.nan

    # current_eps = price/per（perが無いなら不可）
    if pd.isna(per) or per <= 0:
        return grade, np.nan, np.nan

    current_eps = current_price / per
    target_price = current_eps * target_per
    upside = (target_price / current_price - 1.0) if target_price > 0 else np.nan
    return grade, target_price, upside


# =========================
# ワーカー
# =========================

@dataclass
class ErrorStats:
    lock: Lock
    recent: List[bool]  # throttle error flags（直近）

    def push(self, is_throttle: bool) -> None:
        with self.lock:
            self.recent.append(is_throttle)
            if len(self.recent) > THROTTLE_WINDOW:
                self.recent.pop(0)

    def should_abort(self) -> bool:
        with self.lock:
            if len(self.recent) < THROTTLE_WINDOW:
                return False
            ratio = sum(1 for x in self.recent if x) / len(self.recent)
            return ratio >= THROTTLE_RATIO


def fetch_one(code: str, name: str, sector: str, estats: ErrorStats) -> Dict[str, Any]:
    _sleep_rand(PER_TICKER_SLEEP_MIN, PER_TICKER_SLEEP_MAX)

    # 初期値（欠損はNaN/None）
    out: Dict[str, Any] = {
        "code": str(code),
        "name": name if name else None,
        "sector": sector if sector else None,
        "price": np.nan,
        "target_price": np.nan,
        "upside": np.nan,
        "grade": None,
        "peg": np.nan,
        "roic": np.nan,
        "final_g": np.nan,
        "psr": np.nan,
        "equity_ratio": np.nan,
        "div_yield": np.nan,
        "div_streak": None,
        "market_cap": np.nan,
        "roe": np.nan,
        "roa": np.nan,
        "op_cf_margin": np.nan,
        "per": np.nan,
        "pbr": np.nan,
        "adj_pbr": np.nan,
        "cash": np.nan,
        "debt": np.nan,
        "net_de_ratio": np.nan,
        "g_raw": np.nan,
        "op_raw": np.nan,
        "earnings_alert": None,
        "earnings_date": None,
        "latest_i_int": np.nan,
        "k_int": np.nan,
    }

    symbol = f"{code}.T"
    try:
        stock = yf.Ticker(symbol)

        # 価格・時価総額（軽め）
        current_price = np.nan
        market_cap = np.nan
        try:
            fi = stock.fast_info
            if hasattr(fi, "last_price"):
                current_price = _to_float(fi.last_price)
            elif isinstance(fi, dict) and "lastPrice" in fi:
                current_price = _to_float(fi["lastPrice"])

            if hasattr(fi, "market_cap"):
                market_cap = _to_float(fi.market_cap)
            elif isinstance(fi, dict) and "marketCap" in fi:
                market_cap = _to_float(fi["marketCap"])
        except Exception:
            pass

        out["price"] = _round_or_nan(current_price, 0)
        out["market_cap"] = _round_or_nan(_safe_div(market_cap, TO_OKU), 0)

        # 財務
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow

        if financials is None or financials.empty or balance_sheet is None or balance_sheet.empty:
            return out

        latest_pl = financials.columns[0]
        latest_bs = balance_sheet.columns[0]

        revenue = _get_value(financials, ["Total Revenue", "Total Sales"], latest_pl)
        op_income = _get_value(financials, ["Operating Income", "Operating Profit"], latest_pl)
        net_income = _get_value(financials, ["Net Income", "Net Income Common Stockholders"], latest_pl)

        total_assets = _get_value(balance_sheet, ["Total Assets"], latest_bs)
        equity = _get_value(balance_sheet, [
            "Total Stockholder Equity",
            "Total Equity",
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest",
        ], latest_bs)

        forex_adj = _get_value(balance_sheet, [
            "Foreign Currency Translation Adjustments",
            "Accumulated Other Comprehensive Income",
            "Other Comprehensive Income",
        ], latest_bs)

        adjusted_equity = equity
        if not pd.isna(equity) and not pd.isna(forex_adj):
            adjusted_equity = equity - forex_adj
            if adjusted_equity <= 0:
                adjusted_equity = equity

        cash_eq = _get_value(balance_sheet, [
            "Cash And Cash Equivalents",
            "Cash",
            "CashCashEquivalentsAndShortTermInvestments",
        ], latest_bs)

        total_debt = _get_value(balance_sheet, ["Total Debt"], latest_bs)
        if not pd.isna(total_debt) and total_debt > 0:
            debt = total_debt
        else:
            short_debt = _get_value(balance_sheet, ["Current Debt", "Short Long Term Debt"], latest_bs)
            long_debt = _get_value(balance_sheet, ["Long Term Debt"], latest_bs)
            debt = (0 if pd.isna(short_debt) else short_debt) + (0 if pd.isna(long_debt) else long_debt)

        net_de_ratio = np.nan
        if not pd.isna(equity) and equity > 0:
            net_de_ratio = (debt - (0 if pd.isna(cash_eq) else cash_eq)) / equity

        out["cash"] = _round_or_nan(_safe_div(cash_eq, TO_OKU), 0)
        out["debt"] = _round_or_nan(_safe_div(debt, TO_OKU), 0)
        out["net_de_ratio"] = _round_or_nan(net_de_ratio, 2)

        # CF（3年平均）
        op_cf_3yr_avg = np.nan
        if cashflow is not None and not cashflow.empty:
            cf_dates = list(cashflow.columns[:3])
            vals = []
            for d in cf_dates:
                v = _get_value(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"], d)
                if not pd.isna(v):
                    vals.append(v)
            if vals:
                op_cf_3yr_avg = float(np.mean(vals))

        # 成長率・g_raw
        growth_dates = list(financials.columns[:6])
        final_g, g_raw, _ = _calc_growth_median(financials, growth_dates)
        out["final_g"] = _round_or_nan(final_g, 1)
        out["g_raw"] = _round_or_nan(g_raw, 1)

        # op_raw（直近前年差）
        op_raw = np.nan
        if len(growth_dates) >= 2:
            op_latest = _get_value(financials, ["Operating Income", "Operating Profit"], growth_dates[0])
            op_prev = _get_value(financials, ["Operating Income", "Operating Profit"], growth_dates[1])
            if not pd.isna(op_prev) and op_prev > 0:
                op_raw = ((op_latest - op_prev) / abs(op_prev)) * 100.0
        out["op_raw"] = _round_or_nan(op_raw, 1)

        # 配当
        div_yield, div_streak = _calc_dividend_metrics(stock, current_price)
        out["div_yield"] = _round_or_nan(div_yield, 2)
        out["div_streak"] = div_streak

        # 指標
        per = np.nan
        eps_latest = _get_eps_for_date(financials, growth_dates[0]) if growth_dates else np.nan
        if not pd.isna(current_price) and current_price > 0 and not pd.isna(eps_latest) and eps_latest > 0:
            per = current_price / eps_latest
        out["per"] = _round_or_nan(per, 1)

        peg = np.nan
        if (not pd.isna(per)) and per > 0 and per <= 200 and (not pd.isna(final_g)) and final_g > 0:
            peg = per / final_g
        out["peg"] = _round_or_nan(peg, 2)

        pbr = np.nan
        if (not pd.isna(market_cap)) and market_cap > 0 and (not pd.isna(equity)) and equity > 0:
            pbr = market_cap / equity
        out["pbr"] = _round_or_nan(pbr, 2)

        psr = np.nan
        if (not pd.isna(market_cap)) and market_cap > 0 and (not pd.isna(revenue)) and revenue > 0:
            psr = market_cap / revenue
        out["psr"] = _round_or_nan(psr, 2)

        equity_ratio = np.nan
        if (not pd.isna(equity)) and (not pd.isna(total_assets)) and total_assets > 0:
            equity_ratio = (equity / total_assets) * 100.0
        out["equity_ratio"] = _round_or_nan(equity_ratio, 1)

        op_cf_margin = np.nan
        if (not pd.isna(op_cf_3yr_avg)) and (not pd.isna(revenue)) and revenue > 0:
            op_cf_margin = (op_cf_3yr_avg / revenue) * 100.0
        out["op_cf_margin"] = _round_or_nan(op_cf_margin, 2)

        # ROIC/ROE/ROA
        tax_rate = 0.30
        roic = np.nan
        if not pd.isna(op_income):
            nopat = op_income * (1.0 - tax_rate)
            invested = debt + adjusted_equity
            if (not pd.isna(invested)) and invested > 0:
                roic = (nopat / invested) * 100.0
        out["roic"] = _round_or_nan(roic, 2)

        roe = np.nan
        if (not pd.isna(net_income)) and (not pd.isna(equity)) and equity > 0:
            roe = (net_income / equity) * 100.0
        out["roe"] = _round_or_nan(roe, 2)

        roa = np.nan
        if (not pd.isna(net_income)) and (not pd.isna(total_assets)) and total_assets > 0:
            roa = (net_income / total_assets) * 100.0
        out["roa"] = _round_or_nan(roa, 2)

        # 無形資産調整
        latest_i_int, k_int, adj_pbr = _calc_intangibles_adj_pbr(financials, balance_sheet, equity, market_cap)
        out["latest_i_int"] = _round_or_nan(_safe_div(latest_i_int, TO_OKU), 0)
        out["k_int"] = _round_or_nan(_safe_div(k_int, TO_OKU), 0)
        out["adj_pbr"] = _round_or_nan(adj_pbr, 2)

        # 決算
        earnings_alert, earnings_date = _calc_earnings(stock)
        out["earnings_alert"] = earnings_alert
        out["earnings_date"] = earnings_date

        # Grade & Target
        grade, target_price, upside = _grade_and_target(
            current_price=current_price,
            per=per,
            roic=roic,
            final_g=final_g,
            psr=psr,
            equity_ratio=equity_ratio,
            net_de_ratio=net_de_ratio,
            div_yield=div_yield,
            div_streak=div_streak,
            op_cf_3yr_avg=op_cf_3yr_avg,
            adj_pbr=adj_pbr,
            pbr=pbr,
        )
        out["grade"] = grade
        out["target_price"] = _round_or_nan(target_price, 0)
        out["upside"] = _round_or_nan(upside, 3)

        return out

    except Exception as e:
        msg = str(e)
        estats.push(_is_throttle_error(msg))
        if estats.should_abort():
            raise RuntimeError(f"THROTTLE_ABORT: throttle errors ratio exceeded ({THROTTLE_RATIO})") from e
        return out


# =========================
# メイン
# =========================

def load_master(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    df["code"] = df["code"].astype(str).str.strip()
    df["name"] = df.get("name", "").astype(str)
    df["sector"] = df.get("sector", "").astype(str)
    df = df[df["code"].str.match(r"^\d{4}$", na=False)].copy()
    df = df.drop_duplicates(subset=["code"], keep="last")
    return df


def enforce_schema(df: pd.DataFrame) -> pa.Table:
    df = df[[f.name for f in SCHEMA]]

    int_cols = ["div_streak"]
    for c in int_cols:
        df[c] = df[c].astype("Int32")  # nullable

    float_cols = [
        "price", "target_price", "upside", "peg", "roic", "final_g", "psr", "equity_ratio", "div_yield",
        "market_cap", "roe", "roa", "op_cf_margin", "per", "pbr", "adj_pbr", "cash", "debt", "net_de_ratio",
        "g_raw", "op_raw", "latest_i_int", "k_int"
    ]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    str_cols = ["code", "name", "sector", "grade", "earnings_alert", "earnings_date"]
    for c in str_cols:
        df[c] = df[c].where(df[c].notna(), None)
        df[c] = df[c].astype("object")

    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)

    meta = dict(table.schema.metadata or {})
    meta.update({
        b"generated_at_utc": _now_utc_iso().encode("utf-8"),
        b"master_csv_url": MASTER_CSV_URL.encode("utf-8"),
    })
    table = table.replace_schema_metadata(meta)
    return table


def validate_and_finalize(tmp_path: str, out_path: str, df: pd.DataFrame) -> None:
    # A 行数
    if len(df) < MIN_ROWS:
        raise RuntimeError(f"VALIDATION_FAIL: rows={len(df)} < {MIN_ROWS}")

    # B サイズ
    if MIN_BYTES > 0:
        size = os.path.getsize(tmp_path)
        if size < MIN_BYTES:
            raise RuntimeError(f"VALIDATION_FAIL: bytes={size} < {MIN_BYTES}")

    # C 欠損率（price）
    na_ratio = float(df["price"].isna().mean())
    if na_ratio > MAX_PRICE_NA_RATIO:
        raise RuntimeError(f"VALIDATION_FAIL: price_na_ratio={na_ratio:.4f} > {MAX_PRICE_NA_RATIO}")

    shutil.move(tmp_path, out_path)


def main() -> int:
    print(f"[INFO] MASTER_CSV_URL={MASTER_CSV_URL}")
    print(f"[INFO] MAX_WORKERS={MAX_WORKERS} CHUNK_SIZE={CHUNK_SIZE}")

    master = load_master(MASTER_CSV_URL)
    print(f"[INFO] universe={len(master)}")

    estats = ErrorStats(lock=Lock(), recent=[])

    results: List[Dict[str, Any]] = []
    codes = master[["code", "name", "sector"]].to_dict("records")

    for i in range(0, len(codes), CHUNK_SIZE):
        chunk = codes[i:i + CHUNK_SIZE]
        print(f"[INFO] chunk {i+1}-{min(i+CHUNK_SIZE, len(codes))} / {len(codes)}")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(fetch_one, x["code"], x["name"], x["sector"], estats) for x in chunk]
            for fut in as_completed(futs):
                results.append(fut.result())

        _sleep_rand(CHUNK_SLEEP_MIN, CHUNK_SLEEP_MAX)

    df = pd.DataFrame(results)

    # 念のため欠損統一（"-" は出さない設計だが保険）
    df = df.replace("-", np.nan)

    # 列不足があれば埋める（事故防止）
    for col in [f.name for f in SCHEMA]:
        if col not in df.columns:
            df[col] = np.nan

    # codeで一意化し、masterの順に並べる（参照側が安定）
    df["code"] = df["code"].astype(str)
    df = df.drop_duplicates(subset=["code"], keep="last")
    df = master[["code"]].merge(df, on="code", how="left")

    # name/sector は master を優先
    df["name"] = master.set_index("code").loc[df["code"], "name"].to_numpy()
    df["sector"] = master.set_index("code").loc[df["code"], "sector"].to_numpy()

    table = enforce_schema(df)
    pq.write_table(table, TMP_NAME, compression="zstd")
    print(f"[INFO] wrote {TMP_NAME} bytes={os.path.getsize(TMP_NAME)} rows={len(df)}")

    validate_and_finalize(TMP_NAME, OUT_NAME, df)
    print(f"[OK] finalized {OUT_NAME} bytes={os.path.getsize(OUT_NAME)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
