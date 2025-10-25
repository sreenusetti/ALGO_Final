#!/usr/bin/env python3
# trading_bot.py
import logging
import os
import json
import time
import sys
import threading
import warnings
import traceback
from csv import DictWriter
import pickle
import datetime as dt
from datetime import time as dt_time
import pandas as pd
import numpy as np
import pytz
try:
    import joblib  # preferred for sklearn pipelines
except ImportError:
    joblib = None
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
from modules.Fyers.service import save_to_json, load_from_json, bollinger_bands, ema, atr
from modules.Fyers.adx_efi_mom.service import fetchOHLC1
from math import isfinite
from cpr_ai_predictor import CPR_AIPredictor
from typing import Optional


warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


# ─────────────────────────────────────────────────────────────────────────────
# GAP PROTECTOR (ORB + Retest + Gap-aware SL/Exit)
# ─────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass as _dataclass_gap
import datetime as _dt_gap

@_dataclass_gap
class GapConfig:
    orb_minutes: int = 15
    gap_small: float = 0.005      # 0.5%
    retest_tol: float = 0.0015    # 0.15% proximity
    buffer_frac: float = 0.001    # 0.1% SL buffer


# =============================================================================
# COMPLETE GAP PROTECTOR CLASS - All Methods Included
# =============================================================================
from dataclasses import dataclass as _dataclass_gap
import datetime as _dt_gap
import pandas as pd


@_dataclass_gap
class GapConfig:
    orb_minutes: int = 15
    gap_small: float = 0.005  # 0.5%
    retest_tol: float = 0.0015  # 0.15% proximity
    buffer_frac: float = 0.001  # 0.1% SL buffer


class GapProtector:
    """
    Complete Gap Protection System:
    - Detects gap type (UP/DOWN/FLAT)
    - Tracks Opening Range Breakout (ORB)
    - Requires retest for outside gaps
    - Provides gap-aware stop loss
    - Monitors gap failure conditions
    """

    def __init__(self, log):
        self.log = log
        self.cfg = GapConfig()
        self.state = {
            "session_date": None,
            "open": None,
            "prev_close": None,
            "pdh": None,  # Previous Day High
            "pdl": None,  # Previous Day Low
            "tc": None,  # Top Central Pivot
            "bc": None,  # Bottom Central Pivot
            "type": "FLAT",  # GAP_UP, GAP_DOWN, or FLAT
            "outside": False,  # Whether gap opened outside PDH/PDL
            "or_high": None,  # Opening Range High
            "or_low": None,  # Opening Range Low
            "orb_frozen": False,  # Whether ORB period has ended
            "retest_done": False  # Whether retest has been completed
        }
        self.log("[GAP] GapProtector initialized", True)

    def _near(self, px, lvl):
        """Check if price is near a level (within tolerance)"""
        if px is None or lvl is None:
            return False
        try:
            return abs(px - lvl) / max(lvl, 1e-9) <= self.cfg.retest_tol
        except Exception:
            return False

    def reset_if_new_session(self, ts, ohlc_df, piv):
        """
        Reset gap state on new trading session

        Args:
            ts: Current timestamp
            ohlc_df: Historical OHLC dataframe (daily)
            piv: Pivot data dictionary with TC, BC, PDH, PDL
        """
        try:
            # Normalize timestamp to IST date
            try:
                ts = ts.astimezone(IST) if hasattr(ts, "astimezone") else ts
            except Exception:
                pass

            day = (ts.date() if hasattr(ts, "date") else None)
            if not day:
                self.log("[GAP] Could not extract date from timestamp", True)
                return

            # Check if we already processed this session
            if self.state.get("session_date") == day:
                return

            self.log(f"[GAP] 🔄 New session detected: {day}", False)

            # Extract today's open price from OHLC dataframe
            open_px = None
            prev_close = None

            # Extract pivot levels
            pdh = float(piv.get("High") or piv.get("PDH") or 0.0) if isinstance(piv, dict) else 0.0
            pdl = float(piv.get("Low") or piv.get("PDL") or 0.0) if isinstance(piv, dict) else 0.0
            tc = float(piv.get("TC") or 0.0) if isinstance(piv, dict) else 0.0
            bc = float(piv.get("BC") or 0.0) if isinstance(piv, dict) else 0.0
            prev_close = float(piv.get("Close") or 0.0) if isinstance(piv, dict) else 0.0

            # Get today's opening price from dataframe
            if ohlc_df is not None and not ohlc_df.empty:
                try:
                    # Convert index to IST if needed
                    if hasattr(ohlc_df.index, "tz_convert"):
                        df_today = ohlc_df[ohlc_df.index.tz_convert(IST).date == day]
                    else:
                        df_today = ohlc_df[ohlc_df.index.date == day]

                    if len(df_today) > 0:
                        open_px = float(df_today.iloc[0]["Open"])
                        self.log(f"[GAP] Today's open: {open_px:.2f}", True)
                except Exception as e:
                    self.log(f"[GAP] Error extracting today's open: {e}", True)

            # Update state with new session data
            self.state.update({
                "session_date": day,
                "open": open_px,
                "prev_close": prev_close,
                "pdh": pdh,
                "pdl": pdl,
                "tc": tc,
                "bc": bc,
                "or_high": None,
                "or_low": None,
                "orb_frozen": False,
                "retest_done": False
            })

            # Classify gap type
            if open_px and prev_close and prev_close > 0:
                gap = (open_px - prev_close) / prev_close

                if gap >= self.cfg.gap_small:
                    self.state["type"] = "GAP_UP"
                    self.log(f"[GAP] 📈 GAP UP detected: {gap * 100:.2f}%", False)
                elif gap <= -self.cfg.gap_small:
                    self.state["type"] = "GAP_DOWN"
                    self.log(f"[GAP] 📉 GAP DOWN detected: {gap * 100:.2f}%", False)
                else:
                    self.state["type"] = "FLAT"
                    self.log(f"[GAP] ➡️ FLAT open: {gap * 100:.2f}%", False)
            else:
                self.state["type"] = "FLAT"
                self.log("[GAP] ⚠️ Could not classify gap (missing open/prev_close)", True)

            # Check if gap opened outside previous day range
            self.state["outside"] = bool(
                (open_px and pdh and open_px > pdh) or
                (open_px and pdl and open_px < pdl)
            )

            if self.state["outside"]:
                self.log(f"[GAP] ⚠️ OUTSIDE GAP: Open {open_px:.2f} outside PDH {pdh:.2f} / PDL {pdl:.2f}", False)

            self.log(
                f"[GAP] Session initialized | Type: {self.state['type']} | "
                f"Outside: {self.state['outside']} | Open: {open_px} | "
                f"Prev Close: {prev_close} | PDH: {pdh} | PDL: {pdl}",
                False
            )

        except Exception as e:
            try:
                self.log(f"[GAP] ❌ reset_if_new_session error: {e}", False)
                import traceback
                self.log(f"[GAP] Traceback: {traceback.format_exc()}", True)
            except Exception:
                pass

    def update_opening_range(self, ts, ohlc_df):
        """
        Update Opening Range (OR) during first N minutes

        Args:
            ts: Current timestamp
            ohlc_df: Intraday OHLC dataframe
        """
        try:
            day = self.state.get("session_date")
            if not day or ohlc_df is None or len(ohlc_df) == 0:
                return

            # Already frozen, no need to update
            if self.state.get("orb_frozen"):
                return

            # Find today's candles
            try:
                if hasattr(ohlc_df.index, "tz_convert"):
                    df_today = ohlc_df[ohlc_df.index.tz_convert(IST).date == day]
                else:
                    df_today = ohlc_df[ohlc_df.index.date == day]
            except Exception as e:
                self.log(f"[GAP] Error filtering today's data: {e}", True)
                return

            if len(df_today) == 0:
                return

            # Calculate elapsed time since market open
            first_ts = df_today.index[0]
            try:
                elapsed = (ts - first_ts).total_seconds()
            except Exception:
                elapsed = 1e9  # Assume frozen if we can't compute

            # Update OR if still within ORB window
            if elapsed < self.cfg.orb_minutes * 60:
                self.state["or_high"] = float(df_today["High"].iloc[:len(df_today)].max())
                self.state["or_low"] = float(df_today["Low"].iloc[:len(df_today)].min())
                self.log(
                    f"[GAP] 📊 ORB update: High={self.state['or_high']:.2f}, "
                    f"Low={self.state['or_low']:.2f} ({elapsed / 60:.1f}/{self.cfg.orb_minutes} min)",
                    True
                )
            else:
                # Freeze ORB
                if not self.state.get("orb_frozen"):
                    self.state["orb_frozen"] = True
                    self.log(
                        f"[GAP] 🔒 ORB FROZEN at {self.cfg.orb_minutes} min: "
                        f"High={self.state['or_high']:.2f}, Low={self.state['or_low']:.2f}",
                        False
                    )

        except Exception as e:
            try:
                self.log(f"[GAP] ❌ update_opening_range error: {e}", False)
            except Exception:
                pass

    def block_entry(self, side, ts, ltp, ema5, ema21):
        """
        Check if entry should be blocked based on gap rules

        Args:
            side: "Buy" or "Sell"
            ts: Current timestamp
            ltp: Last traded price
            ema5: 5-period EMA
            ema21: 21-period EMA

        Returns:
            (blocked: bool, reason: str, sl_override: float|None)
        """
        st = self.state
        if side not in ("Buy", "Sell"):
            return (False, "", None)

        self.log(
            f"[GAP-CHECK] Entry: side={side}, type={st.get('type')}, "
            f"orb_frozen={st.get('orb_frozen')}, outside={st.get('outside')}",
            True
        )

        # ==========================================
        # PHASE 1: During ORB (First 15 minutes)
        # ==========================================
        if not st.get("orb_frozen"):
            if st.get("type") == "GAP_UP":
                if side == "Sell":
                    self.log("[GAP-BLOCK] ❌ ORB window on gap-up – no shorts", False)
                    return (True, "ORB window on gap-up – no shorts", None)

                # For BUY: Need ORH breakout + EMA5>21 + >TC
                if st.get("or_high") and ltp is not None and ema5 is not None and ema21 is not None:
                    orh_break = ltp > st["or_high"]
                    ema_bull = float(ema5) >= float(ema21)
                    tc_clear = (st.get("tc") is None or ltp > st.get("tc", 0))

                    if not (orh_break and ema_bull and tc_clear):
                        reason = (
                            f"Need ORH breakout + EMA5>21 + >TC "
                            f"(ORH={st['or_high']:.2f}, LTP={ltp:.2f}, "
                            f"EMA5={ema5:.2f}, EMA21={ema21:.2f}, TC={st.get('tc', 0):.2f})"
                        )
                        self.log(f"[GAP-BLOCK] ❌ {reason}", False)
                        return (True, reason, None)

            elif st.get("type") == "GAP_DOWN":
                if side == "Buy":
                    self.log("[GAP-BLOCK] ❌ ORB window on gap-down – no longs", False)
                    return (True, "ORB window on gap-down – no longs", None)

                # For SELL: Need ORL breakdown + EMA5<21 + <BC
                if st.get("or_low") and ltp is not None and ema5 is not None and ema21 is not None:
                    orl_break = ltp < st["or_low"]
                    ema_bear = float(ema5) <= float(ema21)
                    bc_clear = (st.get("bc") is None or ltp < st.get("bc", 9e9))

                    if not (orl_break and ema_bear and bc_clear):
                        reason = (
                            f"Need ORL breakdown + EMA5<21 + <BC "
                            f"(ORL={st['or_low']:.2f}, LTP={ltp:.2f}, "
                            f"EMA5={ema5:.2f}, EMA21={ema21:.2f}, BC={st.get('bc', 0):.2f})"
                        )
                        self.log(f"[GAP-BLOCK] ❌ {reason}", False)
                        return (True, reason, None)

            self.log("[GAP-CHECK] ✅ ORB window passed", True)
            return (False, "", None)

        # ==========================================
        # PHASE 2: After ORB - Retest Requirement
        # ==========================================
        if st.get("outside") and not st.get("retest_done"):
            if st.get("type") == "GAP_UP":
                # Check if price has retested PDH or TC
                if self._near(ltp, st.get("pdh")) or self._near(ltp, st.get("tc")):
                    st["retest_done"] = True
                    self.log(
                        f"[GAP] ✅ Retest complete at {ltp:.2f} "
                        f"(PDH={st.get('pdh'):.2f}, TC={st.get('tc'):.2f})",
                        False
                    )
                else:
                    reason = (
                        f"Wait retest PDH/TC on gap-up "
                        f"(LTP={ltp:.2f}, PDH={st.get('pdh'):.2f}, TC={st.get('tc'):.2f})"
                    )
                    self.log(f"[GAP-BLOCK] ⏳ {reason}", False)
                    return (True, reason, None)

            elif st.get("type") == "GAP_DOWN":
                # Check if price has retested PDL or BC
                if self._near(ltp, st.get("pdl")) or self._near(ltp, st.get("bc")):
                    st["retest_done"] = True
                    self.log(
                        f"[GAP] ✅ Retest complete at {ltp:.2f} "
                        f"(PDL={st.get('pdl'):.2f}, BC={st.get('bc'):.2f})",
                        False
                    )
                else:
                    reason = (
                        f"Wait retest PDL/BC on gap-down "
                        f"(LTP={ltp:.2f}, PDL={st.get('pdl'):.2f}, BC={st.get('bc'):.2f})"
                    )
                    self.log(f"[GAP-BLOCK] ⏳ {reason}", False)
                    return (True, reason, None)

        # ==========================================
        # PHASE 3: Gap-Aware Stop Loss Suggestion
        # ==========================================
        sl = None
        try:
            if side == "Buy" and st.get("type") == "GAP_UP":
                # For longs on gap-up, use max of PDH/TC/BC as base
                base = max([x for x in [st.get("pdh"), st.get("tc"), st.get("bc")] if x])
                sl = base * (1 - self.cfg.buffer_frac) if base else None

            elif side == "Sell" and st.get("type") == "GAP_DOWN":
                # For shorts on gap-down, use min of PDL/BC/TC as base
                base = min([x for x in [st.get("pdl"), st.get("bc"), st.get("tc")] if x])
                sl = base * (1 + self.cfg.buffer_frac) if base else None
        except Exception as e:
            self.log(f"[GAP] SL calc error: {e}", True)
            sl = None

        if sl:
            self.log(f"[GAP] 🎯 Dynamic SL suggested: {sl:.2f}", True)

        return (False, "", sl)

    def gap_fail_exit(self, position_side, close_price):
        """
        Check if gap has failed (price closes back inside previous day's range)

        Args:
            position_side: "LONG" or "SHORT"
            close_price: Current close price

        Returns:
            (should_exit: bool, reason: str)
        """
        st = self.state
        try:
            # GAP_UP failure: Long position closes below PDH
            if position_side == "LONG" and st.get("type") == "GAP_UP":
                if st.get("pdh") and close_price is not None and close_price < st["pdh"]:
                    msg = f"Gap-up failed (close {close_price:.2f} < PDH {st['pdh']:.2f})"
                    self.log(f"[GAP-FAIL] 🔴 {msg}", False)
                    return True, msg

            # GAP_DOWN failure: Short position closes above PDL
            if position_side == "SHORT" and st.get("type") == "GAP_DOWN":
                if st.get("pdl") and close_price is not None and close_price > st["pdl"]:
                    msg = f"Gap-down failed (close {close_price:.2f} > PDL {st['pdl']:.2f})"
                    self.log(f"[GAP-FAIL] 🔴 {msg}", False)
                    return True, msg

        except Exception as e:
            self.log(f"[GAP] fail check error: {e}", False)

        return False, ""

    def get_state(self):
        """Get current gap state for debugging"""
        return {
            "session_date": str(self.state.get("session_date")),
            "gap_type": self.state.get("type"),
            "outside": self.state.get("outside"),
            "orb_frozen": self.state.get("orb_frozen"),
            "retest_done": self.state.get("retest_done"),
            "or_high": self.state.get("or_high"),
            "or_low": self.state.get("or_low"),
            "pdh": self.state.get("pdh"),
            "pdl": self.state.get("pdl"),
            "open": self.state.get("open"),
            "prev_close": self.state.get("prev_close")
        }


# =============================================================================
# OBX v3.3 OPTION BUYER — Minimal Integration (self-contained)
# =============================================================================
import re as _re_obx, json as _json_obx, time as _time_obx, math as _math_obx, datetime as _dt_obx
from dataclasses import dataclass as _dataclass_obx
from typing import Dict as _Dict_obx, List as _List_obx, Optional as _Optional_obx, Tuple as _Tuple_obx

try:
    import numpy as _np_obx
except ImportError:
    class _NP_OBX:
        @staticmethod
        def mean(x): return sum(x)/max(1,len(x))
        @staticmethod
        def std(x, ddof=1):
            n=len(x); return 0.0 if n<=1 else (_math_obx.sqrt(sum((xi-sum(x)/n)**2 for xi in x)/max(1,n-ddof)))
        @staticmethod
        def isfinite(x):
            try: return _math_obx.isfinite(float(x))
            except: return False
    _np_obx = _NP_OBX()

# Keep IST in sync with main file timezone (set later); we'll define a local fallback here
try:
    IST  # type: ignore
except NameError:
    IST = _dt_obx.timezone(_dt_obx.timedelta(hours=5, minutes=30))

def _now_ist_obx() -> _dt_obx.datetime:
    try:
        return _dt_obx.datetime.now(IST)
    except Exception:
        return _dt_obx.datetime.now(_dt_obx.timezone(_dt_obx.timedelta(hours=5, minutes=30)))

def _zscore_obx(series: _List_obx[float], last_val: float, lookback: int = 20) -> float:
    if not series: return 0.0
    tail = series[-lookback:]
    mu = _np_obx.mean(tail)
    sd = _np_obx.std(tail, ddof=1) if len(tail) > 1 else 0.0
    return 0.0 if not sd else (last_val - mu) / sd

def _pct_change_obx(now: float, prev: float) -> float:
    return 0.0 if (prev is None or prev == 0) else 100.0 * (now - prev) / prev

@_dataclass_obx
class OptionRules:
    adx_min: float; fr_max: float; dte_min: int
    delta_min: float; delta_max: float; spread_max: float
    atr_sl_mult: float; trail_mult: float
    target_pct: float = 0.25; tsr_min: float = 1.2

@_dataclass_obx
class InstrumentConfig:
    oc_symbol: str; underlying_quote_symbol: str; rules: OptionRules

_PRESETS_OBX: _Dict_obx[str, InstrumentConfig] = {
    "NATGASMINI": InstrumentConfig("MCX:NATGASMINI", "MCX:NATGASMINI-INDEX", OptionRules(22,0.70,3,0.45,0.65,0.005,0.7,3.0)),
    "CRUDEOIL":   InstrumentConfig("MCX:CRUDEOIL",   "MCX:CRUDEOIL-INDEX",   OptionRules(20,0.90,2,0.35,0.55,0.006,0.6,2.5)),
    "_DEFAULT":   InstrumentConfig("", "", OptionRules(18,0.90,2,0.35,0.55,0.006,0.6,2.5)),
}

class SymbolParser:
    MONTHS = {m:i for i,m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"],1)}
    @staticmethod
    def is_option(sym: str) -> bool:
        s = (sym or "").upper()
        return s.endswith("CE") or s.endswith("PE")
    @staticmethod
    def parse(sym: str):
        s = (sym or "").upper()
        right = "CE" if s.endswith("CE") else ("PE" if s.endswith("PE") else None)
        core = s[:-2] if right else s
        m = _re_obx.match(r"^([A-Z]+)(\d{1,2})([A-Z]{3})(\d{2,4})(\d+)$", core)
        if not m: return {"root": core, "right": right, "strike": None, "expiry": None}
        root, dd, mon, yr, k = m.groups()
        day, year = int(dd), int(yr)
        year = 2000 + year if year < 100 else year
        mon_num = SymbolParser.MONTHS.get(mon, 1)
        try: exp = _dt_obx.datetime(year, mon_num, day, 23, 59, tzinfo=IST)
        except: exp = None
        return {"root": root, "right": right, "strike": float(k), "expiry": exp}

class FlowCache:
    def __init__(self, path: str = "./.obx_flow_cache.json"):
        self.path = path
        self.db: _Dict_obx[str, _Dict_obx[str, _List_obx[_Tuple_obx[float,float,float,float]]]] = {}
        self._load()
    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r") as f: self.db = _json_obx.load(f)
        except Exception:
            self.db = {}
    def _save(self):
        try:
            with open(self.path, "w") as f: _json_obx.dump(self.db, f)
        except Exception:
            pass
    def add_point(self, key: str, side: str, ts: float, oi: float, ltp: float, vol: float, keep: int = 200):
        self.db.setdefault(key, {}).setdefault(side, [])
        self.db[key][side].append((ts, float(oi or 0), float(ltp or 0), float(vol or 0)))
        if len(self.db[key][side]) > keep: self.db[key][side] = self.db[key][side][-keep:]
        self._save()
    def last_n(self, key: str, side: str, n: int = 20) -> _List_obx[_Tuple_obx[float,float,float,float]]:
        return (self.db.get(key, {}).get(side, []) or [])[-n:]

def normalize_option_chain(resp: dict):
    d = resp.get("d") or resp.get("data") or resp
    underlying = d.get("underlying_value") or d.get("ltp") or 0.0
    rows: _List_obx[dict] = []
    oc = d.get("optionChain") or d.get("option_chain")
    if oc is None:
        for ed in (d.get("expiryData") or []):
            if "optionChain" in ed:
                oc = ed["optionChain"]; underlying = ed.get("underlying_value") or underlying; break
    if oc:
        for r in oc:
            strike = r.get("strike") or r.get("strikePrice") or r.get("strike_price")
            if strike is None: continue
            row = {
                "strike": float(strike),
                "call": {
                    "ltp": r.get("c_ltp") or r.get("call_ltp") or r.get("ce_ltp") or r.get("call",{}).get("ltp"),
                    "oi":  r.get("c_oi")  or r.get("call_oi") or r.get("ce_oi")  or r.get("call",{}).get("oi"),
                    "d_oi":r.get("c_chg_oi") or r.get("call_chg_oi") or r.get("call",{}).get("chg_oi")
                },
                "put": {
                    "ltp": r.get("p_ltp") or r.get("put_ltp") or r.get("pe_ltp") or r.get("put",{}).get("ltp"),
                    "oi":  r.get("p_oi")  or r.get("put_oi") or r.get("pe_oi")  or r.get("put",{}).get("oi"),
                    "d_oi":r.get("p_chg_oi") or r.get("put_chg_oi") or r.get("put",{}).get("chg_oi")
                }
            }
            rows.append(row)
    return float(underlying or 0.0), rows

@_dataclass_obx
class PositionState:
    entry_price: float; qty: int; side: str; target_px: float; open_time: float; product: str = "INTRADAY"

class OptionBuyingStrategy:
    def __init__(self, fyers_sdk, logger=print, flow_path="./.obx_flow_cache.json"):
        self.fyers_sdk = fyers_sdk
        self.log = logger
        self.flow = FlowCache(flow_path)
        self.position: _Optional_obx[PositionState] = None
        self.order_manager = None  # 🔥 NEW: Link to OrderManager

    @staticmethod
    def _f(x, alt=None):
        try:
            xf = float(x)
            return xf if _np_obx.isfinite(xf) else alt
        except Exception:
            return alt

    @staticmethod
    def _hours_left(expiry): 
        return max(0.25, (expiry - _now_ist_obx()).total_seconds()/3600.0) if expiry else 6.25

    @staticmethod
    def _spread_ok(bid, ask, limit_frac):
        if not bid or not ask or bid <= 0 or ask < bid: return True
        mid = 0.5*(bid+ask); spr = (ask - bid)/max(mid, 1e-6)
        return spr <= limit_frac

    def _rules_for(self, root): 
        cfg = _PRESETS_OBX.get(root, _PRESETS_OBX["_DEFAULT"])
        return cfg.rules, cfg

    def _secret_flow_ok(self, root, row, side):
        k = f"{root}:{int(row['strike'])}"
        hist = self.flow.last_n(k, side, n=20)
        if len(hist) < 5: return True
        _, oi_prev, ltp_prev, _ = hist[-4]; _, oi_now, ltp_now, _ = hist[-1]
        dOI_pct = _pct_change_obx(oi_now, oi_prev); dP_pct = _pct_change_obx(ltp_now, ltp_prev)
        zOI = _zscore_obx([h[1] for h in hist], oi_now); zP = _zscore_obx([h[2] for h in hist], ltp_now)
        return (zOI <= -0.8 and zP >= 0.8 and dOI_pct <= 0 and dP_pct >= 0)

    def _ladder_confirms(self, ladder, atm_idx, side):
        lo, hi = max(0, atm_idx-3), min(len(ladder), atm_idx+4)
        def _d(r): 
            return r["call"]["d_oi"] if side == "CE" else r["put"]["d_oi"]
        net = sum(float(_d(r)) for r in ladder[lo:hi] if _d(r) is not None)
        if net > 0: return False
        itm_slice = ladder[:atm_idx] if side == "CE" else ladder[atm_idx+1:]
        itm_doi = sum(float(_d(r)) for r in itm_slice if _d(r) is not None)
        return itm_doi <= 0

    def _feasibility_tsr(self, premium, delta, theta_day, iv, rules, hours_left):
        RM = (rules.target_pct * premium) / max(delta or 0.05, 0.05)
        FM = 0.0
        if iv and iv > 0:
            FM = 1.0 * (iv/100.0) * _math_obx.sqrt(max(hours_left, 0.25)/6.25) * 100
        FR = RM / max(FM, 1e-6) if FM > 0 else 0.0
        FR_ok = (FM > 0 and FR <= rules.fr_max)
        TSR_ok, TSR = True, 0.0
        if theta_day is not None and theta_day != 0:
            fm_hr = FM / max(hours_left, 0.25)
            TSR = (max(delta or 0.05, 0.05) * fm_hr) / (abs(theta_day)/24.0)
            TSR_ok = TSR >= rules.tsr_min
        if delta is None or iv is None: FR_ok = True
        if theta_day is None: TSR_ok = True
        return FR_ok, TSR_ok, FR, TSR

    def run_tick(self, option_symbol: str, lot_qty: int = 1):
        try:
            # Validate it's an option
            if not SymbolParser.is_option(option_symbol):
                self.log(f"[OBX] Not an option: {option_symbol}")
                return

            # Parse option details
            opt = SymbolParser.parse(option_symbol)
            root = (opt["root"] or "").upper()
            side = opt["right"]  # CE or PE
            expiry = opt["expiry"]
            strike = opt["strike"]

            # Get OBX rules for this underlying
            rules, inst = self._rules_for(root)

            # ========================================
            # GATE 1: Days to Expiry (DTE)
            # ========================================
            if expiry:
                dte = (expiry - _now_ist_obx()).total_seconds() / 86400.0
                if dte < rules.dte_min:
                    self.log(f"[OBX] ❌ Skip: DTE {dte:.2f} < {rules.dte_min} days", True)
                    return  # Falls through to unified strategy

            # ========================================
            # GATE 2: Option Quotes & Spread
            # ========================================
            resp = call_with_rate_limit_retry(self.fyers_sdk.quotes, {"symbols": option_symbol})
            qmap = {}
            if resp and resp.get("s") == "ok":
                for d in resp.get("d", []):
                    qmap[(d.get("n") or "").upper()] = d.get("v", {})

            q = qmap.get(option_symbol.upper(), {})
            ltp_opt = self._f(q.get("lp"))
            bid = self._f(q.get("bid")) or self._f(q.get("best_bid"))
            ask = self._f(q.get("ask")) or self._f(q.get("best_ask"))

            if not self._spread_ok(bid, ask, rules.spread_max):
                self.log(f"[OBX] ❌ Skip: Spread too wide (bid={bid}, ask={ask})", True)
                return

            # ========================================
            # GATE 3: Greeks Validation
            # ========================================
            greeks = q.get("opGreeks") or q.get("greeks") or {}
            delta = self._f(greeks.get("delta"))
            theta_day = self._f(greeks.get("theta"))
            iv = self._f(greeks.get("iv")) or self._f(q.get("iv"))

            if delta is not None and not (rules.delta_min <= abs(delta) <= rules.delta_max):
                self.log(f"[OBX] ❌ Skip: Δ {delta:.2f} outside [{rules.delta_min}, {rules.delta_max}]", True)
                return

            # ========================================
            # GATE 4: Secret OI Flow Check
            # ========================================
            # Get option chain for OI analysis
            icfg = inst if inst.oc_symbol else _PRESETS_OBX.get(root, _PRESETS_OBX["_DEFAULT"])
            oc_resp = call_with_rate_limit_retry(
                self.fyers_sdk.optionchain,
                {"symbol": icfg.oc_symbol, "strikecount": 7}
            ) or {}

            uval, ladder = normalize_option_chain(oc_resp)
            if not ladder:
                self.log("[OBX] ❌ Skip: Empty option chain", True)
                return

            # Store flow data for analysis
            ts = _time_obx.time()
            for r in ladder:
                k = f"{root}:{int(r['strike'])}"
                if r["call"]["oi"] or r["call"]["ltp"]:
                    self.flow.add_point(k, "CE", ts, float(r["call"]["oi"] or 0),
                                        float(r["call"]["ltp"] or 0), 0.0)
                if r["put"]["oi"] or r["put"]["ltp"]:
                    self.flow.add_point(k, "PE", ts, float(r["put"]["oi"] or 0),
                                        float(r["put"]["ltp"] or 0), 0.0)

            # Find the strike we're trading
            atm_row = min(ladder, key=lambda x: abs((x["strike"] or 0) - (uval or 0)))
            atm_idx = ladder.index(atm_row)
            window = ladder[max(0, atm_idx - 1):atm_idx + 2]
            row = min(window, key=lambda x: abs(x["strike"] - strike)) if strike else atm_row

            # Check secret flow (OI↓ + LTP↑)
            if not self._secret_flow_ok(root, row, side):
                self.log("[OBX] ❌ Skip: Secret flow NOT confirmed (need OI↓ + LTP↑)", True)
                return

            # Check ladder confirmation (ITM OI building)
            if not self._ladder_confirms(ladder, atm_idx, side):
                self.log("[OBX] ❌ Skip: Ladder/CoM did NOT confirm", True)
                return

            # ========================================
            # GATE 5: Feasibility (FR & TSR)
            # ========================================
            if ltp_opt is None or ltp_opt <= 0:
                self.log("[OBX] ❌ Skip: Missing option LTP", True)
                return

            FR_ok, TSR_ok, FR, TSR = self._feasibility_tsr(
                ltp_opt, abs(delta) if delta else None, theta_day, iv,
                rules, self._hours_left(expiry)
            )

            if not FR_ok:
                self.log(f"[OBX] ❌ Skip: FR {FR:.2f} > {rules.fr_max:.2f}", True)
                return
            if not TSR_ok:
                self.log(f"[OBX] ❌ Skip: TSR {TSR:.2f} < {rules.tsr_min:.2f}", True)
                return

            # ========================================
            # ✅ ALL GATES PASSED - PLACE ORDER
            # ========================================
            self.log(f"[OBX] ✅ ALL GATES PASSED - Placing order", False)
            self.log(f"[OBX] Entry: {option_symbol} | Δ={delta:.2f} | IV={iv:.1f}% | FR={FR:.2f} | TSR={TSR:.2f}",
                     False)

            if self.position is None:
                qty = max(1, int(lot_qty))
                payload = {
                    "symbol": option_symbol, "qty": qty, "type": 2, "side": 1,
                    "productType": "INTRADAY", "validity": "DAY",
                    "disclosedQty": 0, "offlineOrder": False
                }
                resp = call_with_rate_limit_retry(self.fyers_sdk.place_order, data=payload) or {}

                if (resp or {}).get("s") != "ok":
                    self.log(f"[OBX] ❌ Order rejected: {resp}", False)
                    return

                # Set target price (25% gain)
                target_px = ltp_opt * (1.0 + rules.target_pct)

                self.position = PositionState(
                    entry_price=ltp_opt, qty=qty, side=side,
                    target_px=target_px, open_time=_time_obx.time()
                )

                self.log(f"[OBX] 🎯 ENTER {option_symbol} @ ₹{ltp_opt:.2f} | Target: ₹{target_px:.2f}", False)

                # 🔥 Update OrderManager state
                if self.order_manager:
                    self.order_manager.position = {
                        "type": "BUY",  # OBX only buys options
                        "entry_price": ltp_opt,
                        "order_id": resp.get("id", f"OBX-{int(_time_obx.time())}"),
                        "ts": _dt_obx.datetime.now(IST).isoformat(),
                        "_obx_managed": True,
                        "stop_loss": ltp_opt * 0.5,  # 50% stop loss for options
                        "r_mult": ltp_opt * 0.5  # For tracking
                    }
                    self.order_manager._save_state()
            else:
                self.log("[OBX] Position already open - managing...", True)

            # Manage existing position
            self._manage_position(option_symbol, rules)

        except Exception as e:
            self.log(f"[OBX] ❌ ERROR: {e}", False)
            import traceback
            self.log(f"[OBX] Traceback: {traceback.format_exc()}", True)

    def _manage_position(self, option_symbol, rules):
        if not self.position: return
        resp = call_with_rate_limit_retry(self.fyers_sdk.quotes, {"symbols": option_symbol}) or {}
        q = {}
        for d in (resp.get("d") or []):
            name = (d.get("n") or "").upper()
            q[name] = d.get("v", {})
        ltp = self._f(q.get(option_symbol.upper(), {}).get("lp"))
        if ltp is None: return
        pos = self.position
        if ltp >= pos.target_px:
            payload = {"symbol": option_symbol, "qty": pos.qty, "type": 2, "side": -1, "productType": "INTRADAY", "validity": "DAY"}
            resp = call_with_rate_limit_retry(self.fyers_sdk.place_order, data=payload) or {}
            self.log(f"[OBX] EXIT TP: {option_symbol} @ {ltp:.2f} | resp={resp}")
            self.position = None
            if self.order_manager:
                self.order_manager.position["type"] = "FLAT"
                self.order_manager._save_state()
            return
        hold_sec = _time_obx.time() - pos.open_time
        if hold_sec > 3*3600:
            payload = {"symbol": option_symbol, "qty": pos.qty, "type": 2, "side": -1, "productType": "INTRADAY", "validity": "DAY"}
            resp = call_with_rate_limit_retry(self.fyers_sdk.place_order, data=payload) or {}
            self.log(f"[OBX] EXIT TIME: {option_symbol} @ {ltp:.2f} | resp={resp}")
            self.position = None
# =============================================================================
# End OBX Integration
# =============================================================================

DEFAULT_TIMEZONE = "Asia/Kolkata"
IST = pytz.timezone(DEFAULT_TIMEZONE)
tf_selected = sys.argv[1] if len(sys.argv) > 1 else "5"
print(f"🕒 Selected timeframe: {tf_selected}m")

AI_GATE_TRADES = True # Set to True to make AI signal a requirement for entry, False to just log it.
logger = logging.getLogger(__name__)
def _build_ai_cpr_features(ltp: float, indicators: dict, pivot_data: dict) -> np.ndarray:
    """
    Builds the feature vector for the AI CPR predictor based on:
    ['plus_di', 'minus_di', 'ema_5', 'ema_21', 'adx', 'rsi', 'efi', 'momentum']
    """
    logger = logging.getLogger(__name__)
    features = []

    def _get_ind_value(key, default_value=0.0):
        val = indicators.get(key, default_value)
        try:
            val = float(val)
            return val if np.isfinite(val) else default_value
        except (ValueError, TypeError):
            return default_value

    # Build features in exact order
    features.append(_get_ind_value("plus_di", 0.0))
    features.append(_get_ind_value("minus_di", 0.0))
    features.append(_get_ind_value("ema_5", ltp))
    features.append(_get_ind_value("ema_21", ltp))
    features.append(_get_ind_value("adx", 0.0))
    features.append(_get_ind_value("rsi", 50.0))
    features.append(_get_ind_value("efi", 0.0))
    features.append(_get_ind_value("momentum", 0.0))

    expected_feature_count = 8
    if len(features) != expected_feature_count:
        logger.error(
            f"[AI-CPR] Feature count mismatch! Expected {expected_feature_count}, got {len(features)}"
        )

    logger.debug(
        f"[AI-CPR] Features built: "
        f"DI+: {features[0]}, DI-: {features[1]}, "
        f"EMA5: {features[2]}, EMA21: {features[3]}, "
        f"ADX: {features[4]}, RSI: {features[5]}, "
        f"EFI: {features[6]}, Momentum: {features[7]}"
    )
    return np.array(features, dtype=float).reshape(1, -1)

# ───────────────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────────────

def call_with_rate_limit_retry(api_func, *args, max_retries=5, **kwargs):
    delay = 5
    for attempt in range(max_retries):
        resp = api_func(*args, **kwargs)
        if isinstance(resp, dict) and (resp.get("code") == 429 or "limit" in str(resp.get("message", "")).lower()):
            print(f"[RATE LIMIT] Rate limit hit. Sleeping for {delay} seconds...")
            time.sleep(delay)
            delay *= 2
        else:
            return resp
    print("[RATE LIMIT] Max retries reached. Skipping this request.")
    return None

def robust_load_json(path, logger_func=print, default=None, debug_only=True):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger_func(f"[DEBUG] Empty JSON: {path}", debug_only)
                    return default
                return json.loads(content)
        logger_func(f"[DEBUG] JSON not found: {path}", debug_only)
        return default
    except json.JSONDecodeError as e:
        logger_func(f"[ERROR] Load JSON decode {path}: {e}", False)
        return default
    except Exception as e:
        logger_func(f"[ERROR] Load JSON {path}: {e}", False)
        return default

def robust_save_json(data, path, logger_func=print, debug_only=True):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger_func(f"[DEBUG] Saved JSON: {path}", debug_only)
        return True
    except Exception as e:
        logger_func(f"[ERROR] Save JSON {path}: {e}", False)
        return False

def convert_to_serializable(obj):
    if isinstance(obj, (bool, np.bool_)): return bool(obj)
    if isinstance(obj, (int, np.integer)): return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (dt.datetime, dt.date, pd.Timestamp)): return obj.isoformat()
    if isinstance(obj, (pd.Series, np.ndarray)):
        return [convert_to_serializable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    if pd.isna(obj): return None
    try:
        return str(obj)
    except Exception as e:
        return None

def convert_dict_to_serializable(d):
    return {str(k): convert_to_serializable(v) for k, v in d.items()}

# ───────────────────────────────────────────────────────────────────────────────
# Market data utils
# ───────────────────────────────────────────────────────────────────────────────

def get_tick_size(fyers_client, symbol):
    """Fetch tick size via fyers.quotes; default to 0.05 on error."""
    try:
        resp = fyers_client.quotes({"symbols": symbol})
        if resp.get("s") == "ok":
            for itm in resp.get("d", []):
                if itm.get("n") == symbol:
                    ts = itm.get("v", {}).get("tick_size")
                    if ts is not None:
                        return float(ts)
        print(f"[WARN] Tick size fetch failed for {symbol}: {resp}")
    except Exception as e:
        print(f"[ERROR] Tick size {symbol}: {e}")
    return 0.05

def vwap(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("VWAP requires DatetimeIndex")
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['CumVol'] = df.groupby(df.index.date)['Volume'].cumsum()
    df['CumPV']  = (tp * df['Volume']).groupby(df.index.date).cumsum()
    df['VWAP']   = df['CumPV'] / df['CumVol']
    df.drop(['CumVol', 'CumPV'], axis=1, inplace=True)
    return df

def adx_efi_mom_trade_signal(df: pd.DataFrame, symbol: str):
    """Calculate ADX, EFI, Momentum, RSI and generate relaxed trade signal."""
    try:
        if df.empty or len(df) < 14:
            return ("NO TRADE",) + (None,)*6
        high, low, close = df['High'], df['Low'], df['Close']

        plus_dm  = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(upper=0)
        tr_cols  = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1)
        tr = tr_cols.max(axis=1)
        atr14 = tr.rolling(14).mean()

        plus_di  = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
        dx       = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx      = dx.rolling(14).mean().iloc[-1]
        di_p, di_m = plus_di.iloc[-1], minus_di.iloc[-1]
        mom    = close.pct_change(10).iloc[-1] * 100 if len(close) >= 10 else None
        efi    = ((close.diff() * df['Volume']).rolling(13).mean()).iloc[-1]
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = -delta.clip(upper=0).rolling(14).mean()
        rs_v   = gain / loss.replace(0, np.nan)
        rsi_v  = (100 - (100/(1+rs_v))).iloc[-1]

        sig = "NO TRADE"
        if adx > 20:
            if di_p > di_m and efi > 0 and mom > 0:
                sig = "BUY"
            elif di_m > di_p and efi < 0 and mom < 0:
                sig = "SELL"

        return sig, float(adx), float(di_p), float(di_m), float(mom), float(efi), float(rsi_v)
    except Exception as e:
        print(f"[ERROR] ADX/EFI/MOM {symbol}: {e}")
        return ("NO TRADE",) + (None,)*6

def fibonacci_retracement(df: pd.DataFrame, period=20, levels=None, logger=print,
                          trend_period=5, proximity_threshold=0.01):
    """Compute Fib levels, trend, behavior & confidence predictions."""
    levels = levels or [0, .236, .382, .5, .618, .786, 1.0]
    if not isinstance(df, pd.DataFrame):
        logger("Fib: df must be DataFrame"); return {}
    for c in ['High','Low','Close','Open']:
        if c not in df.columns:
            logger(f"Fib: missing {c}"); return {}
    if len(df) < period:
        logger(f"Fib: need {period} rows, got {len(df)}"); return {}

    window = df.rolling(window=period)
    high = window['High'].max().iloc[-1]
    low  = window['Low'].min().iloc[-1]
    if pd.isna(high) or pd.isna(low) or high <= low:
        logger("Fib: invalid high/low"); return {}

    diff = high - low
    fibs = {f"Fib_{lvl*100:.1f}%": high - lvl*diff for lvl in levels}

    df2 = df.copy()
    df2['EMA5']  = df2['Close'].ewm(span=trend_period).mean()
    df2['EMA21'] = df2['Close'].ewm(span=21).mean()
    latest = df2.iloc[-1]
    trend = ('Bullish' if latest['EMA5'] > latest['EMA21']
             else 'Bearish' if latest['EMA5'] < latest['EMA21']
             else 'Neutral')

    threshold = diff * proximity_threshold
    respect = {}
    for k, v in fibs.items():
        near = df2[(df2['Low'].between(v-threshold, v+threshold)) |
                   (df2['High'].between(v-threshold, v+threshold))]
        respect[k] = min(len(near)/10, 1.0)

    beh, preds = {}, {}
    price = latest['Close']
    for k, v in fibs.items():
        dist = abs(price - v)/diff
        prox_score = max(0, 1-dist*5)
        resp_score = respect[k]
        if trend == 'Bullish':
            beh[k] = f"Likely support; bounce if reaches {v:.2f}"
            sc = (prox_score*0.5 + resp_score*0.3 + 0.2)*0.9
            rc = (1-prox_score)*0.3
        elif trend == 'Bearish':
            beh[k] = f"Likely resistance; reverse at {v:.2f}"
            sc = (1-prox_score)*0.3
            rc = (prox_score*0.5 + resp_score*0.3 + 0.2)*0.9
        else:
            beh[k] = f"Neutral; watch {v:.2f}"
            sc = (prox_score*0.4 + resp_score*0.2)*0.6
            rc = sc

        preds[k] = {
            'support_confidence': min(max(sc, 0), 1),
            'resistance_confidence': min(max(rc, 0), 1)
        }

    return {'levels': fibs, 'trend': trend, 'behavior': beh, 'predictions': preds}

def rsi_divergence(df: pd.DataFrame, rsi_period=14, lookback=5):
    df2 = df.copy()
    if len(df2) < rsi_period + lookback:
        return False, False

    delta = df2['Close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.rolling(rsi_period).mean()
    avg_l = loss.rolling(rsi_period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df2['RSI'] = 100 - (100/(1+rs))

    df2['Ph'] = df2['High'].rolling(lookback).max()
    df2['Pl'] = df2['Low'].rolling(lookback).min()
    df2['Rh'] = df2['RSI'].rolling(lookback).max()
    df2['Rl'] = df2['RSI'].rolling(lookback).min()

    bullish = ((df2['Low'].iloc[-1] <= df2['Pl'].shift(1).iloc[-1]) and
               (df2['RSI'].iloc[-1] > df2['Rl'].shift(1).iloc[-1]))
    bearish = ((df2['High'].iloc[-1] >= df2['Ph'].shift(1).iloc[-1]) and
               (df2['RSI'].iloc[-1] < df2['Rh'].shift(1).iloc[-1]))
    return bool(bullish), bool(bearish)

def supertrend(df: pd.DataFrame, period=7, multiplier=3, ema5=None, ema21=None, super_guppy=None):
    df2 = df.copy()
    df2['ATR'] = atr(df2, period)
    hl2 = (df2['High'] + df2['Low']) / 2
    df2['BasicUp']   = hl2 + multiplier * df2['ATR']
    df2['BasicDown'] = hl2 - multiplier * df2['ATR']
    df2['FinalUp']   = df2['BasicUp'].copy()
    df2['FinalDown'] = df2['BasicDown'].copy()
    df2['Strend']    = np.nan
    df2['Trend']     = 0

    ind = df2.index
    for i in range(period, len(df2)):
        prev, cur = ind[i-1], ind[i]
        if df2['Close'].iloc[i-1] <= df2['FinalUp'].iloc[i-1]:
            df2.at[cur,'FinalUp'] = min(df2['BasicUp'].iloc[i], df2['FinalUp'].iloc[i-1])
        else:
            df2.at[cur,'FinalUp'] = df2['BasicUp'].iloc[i]
        if df2['Close'].iloc[i-1] >= df2['FinalDown'].iloc[i-1]:
            df2.at[cur,'FinalDown'] = max(df2['BasicDown'].iloc[i], df2['FinalDown'].iloc[i-1])
        else:
            df2.at[cur,'FinalDown'] = df2['BasicDown'].iloc[i]

    flip = None
    for i in range(period, len(df2)):
        prev, cur = ind[i-1], ind[i]
        if df2['Close'].iloc[i-1] <= df2['FinalUp'].iloc[i-1] and df2['Close'].iloc[i] > df2['FinalUp'].iloc[i]:
            df2.at[cur,'Strend'], df2.at[cur,'Trend'] = df2['FinalDown'].iloc[i], 1
            flip = i; break
        if df2['Close'].iloc[i-1] >= df2['FinalDown'].iloc[i-1] and df2['Close'].iloc[i] < df2['FinalDown'].iloc[i]:
            df2.at[cur,'Strend'], df2.at[cur,'Trend'] = df2['FinalUp'].iloc[i], -1
            flip = i; break
    if flip is None:
        flip = period

    for i in range(flip+1, len(df2)):
        prev, cur = ind[i-1], ind[i]
        if df2.at[prev,'Strend'] == df2.at[prev,'FinalUp']:
            if df2['Close'].iloc[i] <= df2['FinalUp'].iloc[i]:
                df2.at[cur,'Strend'], df2.at[cur,'Trend'] = df2['FinalUp'].iloc[i], -1
            else:
                df2.at[cur,'Strend'], df2.at[cur,'Trend'] = df2['FinalDown'].iloc[i], 1
        else:
            if df2['Close'].iloc[i] >= df2['FinalDown'].iloc[i]:
                df2.at[cur,'Strend'], df2.at[cur,'Trend'] = df2['FinalDown'].iloc[i], 1
            else:
                df2.at[cur,'Strend'], df2.at[cur,'Trend'] = df2['FinalUp'].iloc[i], -1

    return df2['Strend'], df2['FinalUp'], df2['FinalDown'], df2['Trend']

# ───────────────────────────────────────────────────────────────────────────────
# EMA50/200 crossover helper
# ───────────────────────────────────────────────────────────────────────────────

class EMA50_200:
    def __init__(self, fyers_client, ticker, interval="30", duration=60):
        self.fyers    = fyers_client
        self.ticker   = ticker
        self.interval = interval
        self.duration = min(duration, 60)
        self.df       = self._fetch_ohlc()

        if not self.df.empty:
            self._validate()
            self._indicators()
            self._signals()
            self._crossovers()

    def _fetch_ohlc(self):
        try:
            today = dt.date.today()
            frm   = (today - dt.timedelta(days=self.duration)).strftime("%Y-%m-%d")
            to    = today.strftime("%Y-%m-%d")
            payload = {
                "symbol": self.ticker,
                "resolution": self.interval,
                "date_format": "1",
                "range_from": frm,
                "range_to": to,
                "cont_flag":"1"
            }
            resp = self.fyers.history(data=payload)
            candles = resp.get("candles", [])
            if not candles:
                print(f"[WARN] No data for {self.ticker}")
                return pd.DataFrame()

            df = pd.DataFrame(candles, columns=["Ts","Open","High","Low","Close","Volume"])
            df["Timestamp"] = pd.to_datetime(df["Ts"], unit="s", utc=True).dt.tz_convert(IST)
            df.set_index("Timestamp", inplace=True)
            return df.sort_index()
        except Exception as e:
            print(f"[ERROR] Fetch OHLC {self.ticker}: {e}")
            return pd.DataFrame()

    def _validate(self):
        missing = [c for c in ['Open','High','Low','Close','Volume'] if c not in self.df]
        if missing:
            raise ValueError(f"Missing cols {missing}")
        if len(self.df) < 200:
            print(f"[WARN] Only {len(self.df)} rows; MA200 may be unreliable")

    def _indicators(self):
        self.df['MA_50']  = self.df['Close'].rolling(50,1).mean()
        self.df['MA_200'] = self.df['Close'].rolling(200,1).mean()

    def _signals(self):
        self.df['Signal']     = np.where(self.df['Close'] > self.df['MA_200'], 'BUY', 'SELL')
        self.df['Distance_%'] = (self.df['Close'] - self.df['MA_200']) / self.df['MA_200'] * 100

    def _crossovers(self):
        self.df['Above'] = self.df['MA_50'] > self.df['MA_200']
        self.df['Cross'] = self.df['Above'].ne(self.df['Above'].shift(1))
        self.df['Type']  = np.where(self.df['MA_50'] > self.df['MA_200'], 'Golden Cross', 'Death Cross')
        self.df['Type']  = self.df['Type'].where(self.df['Cross'], np.nan)
        self.df['Trend'] = np.where(self.df['MA_50'] > self.df['MA_200'], 'Bullish', 'Bearish')

    def get_current_signal(self):
        latest = self.df.iloc[-1]
        return {
            'timestamp': latest.name.strftime('%Y-%m-%d %H:%M:%S'),
            'price': round(latest['Close'], 2),
            'ma_50': round(latest['MA_50'], 2),
            'ma_200': round(latest['MA_200'], 2),
            'signal': latest['Signal'],
            'trend_strength': latest['Distance_%'],
            'trend': latest['Trend'],
            'last_crossover': self._last_crossover(),
            'timeframe': self.interval
        }

    def _last_crossover(self):
        cs = self.df[self.df['Cross']]
        if not cs.empty:
            last = cs.iloc[-1]
            return {
                'date': last.name.strftime('%Y-%m-%d'),
                'timestamp': last.name.strftime('%Y-%m-%d %H:%M:%S'),
                'type': last['Type'],
                'price': round(last['Close'], 2),
                'bars_since': len(self.df) - self.df.index.get_loc(last.name)
            }
        return None

# ─────────────────────────────────
# Ultimate MA (Super Guppy wrapper)
# ─────────────────────────────────

class UltimateMAIndicator:
    def __init__(self, df, params):
        self.df     = df.copy()
        self.params = params

    def calculate(self):
        df = self.df.copy()
        df["ema5"] = df["Close"].ewm(span=5, adjust=False).mean()

        len1 = self.params.get("len", 13)
        len2 = self.params.get("len2", 34)
        df["ma1"] = df["Close"].ewm(span=len1, adjust=False).mean()
        df["ma2"] = df["Close"].ewm(span=len2, adjust=False).mean()

        df["ma_up"]   = df["ma1"] > df["ma2"]
        df["ma_down"] = df["ma1"] < df["ma2"]

        prev_up   = df["ma_up"].shift(1).eq(True)
        prev_down = df["ma_down"].shift(1).eq(True)

        df["cross_up"]   = df["ma_up"] & ~prev_up
        df["cross_down"] = df["ma_down"] & ~prev_down

        df["cr_up"]    = (df["Close"] > df["ma1"]) & ~(df["Close"].shift(1) > df["ma1"].shift(1))
        df["cr_down"]  = (df["Close"] < df["ma1"]) & ~(df["Close"].shift(1) < df["ma1"].shift(1))
        df["cr_up2"]   = (df["Close"] > df["ma2"]) & ~(df["Close"].shift(1) > df["ma2"].shift(1))
        df["cr_down2"] = (df["Close"] < df["ma2"]) & ~(df["Close"].shift(1) < df["ma2"].shift(1))

        df["ema5_above_ma1"] = df["ema5"] > df["ma1"]
        df["ema5_below_ma1"] = df["ema5"] < df["ma1"]

        prev_ema5_up   = df["ema5_above_ma1"].shift(1).eq(True)
        prev_ema5_down = df["ema5_below_ma1"].shift(1).eq(True)

        df["ema5_cross_up"]   = df["ema5_above_ma1"] & ~prev_ema5_up
        df["ema5_cross_down"] = df["ema5_below_ma1"] & ~prev_ema5_down

        return df

    def summarized(self):
        last = self.calculate().iloc[-1]
        return {
            "ma1_value": float(last["ma1"]),
            "ma2_value": float(last["ma2"]),
            "green": bool(last["ma_up"]),
            "red":   bool(last["ma_down"]),
            "cross_up": bool(last["cross_up"]),
            "cross_down": bool(last["cross_down"]),
            "price_cross_ma1_up":   bool(last["cr_up"]),
            "price_cross_ma1_down": bool(last["cr_down"]),
            "price_cross_ma2_up":   bool(last["cr_up2"]),
            "price_cross_ma2_down": bool(last["cr_down2"]),
            "ema5_above_ma1": bool(last["ema5_above_ma1"]),
            "ema5_below_ma1": bool(last["ema5_below_ma1"]),
            "ema5_cross_up":  bool(last["ema5_cross_up"]),
            "ema5_cross_down":bool(last["ema5_cross_down"]),
        }

# ────────────────────────
# Candlestick patterns
# ────────────────────────

class CandlestickAnalyzer:
    bullish_patterns    = ['Hammer','BullishMarubozu','BullishEngulfing','MorningStar',
                           'PiercingLine','BullishThreeLineStrike','BullishKicker']
    bearish_patterns    = ['ShootingStar','BearishMarubozu','BearishEngulfing','EveningStar',
                           'DarkPoolCover','BearishThreeLineStrike','BearishKicker']
    indecision_patterns = ['Doji','SpinningTop','HaramiCross']

    def __init__(self, bot):
        self.bot = bot

    def detect_all(self, df: pd.DataFrame, buffer=0.25):
        df2 = df.copy()
        rng  = df2['High'] - df2['Low']
        body = (df2['Open'] - df2['Close']).abs()
        lower= df2[['Open','Close']].min(axis=1) - df2['Low']
        upper= df2['High'] - df2[['Open','Close']].max(axis=1)

        df2['Doji']            = body <= 0.1*rng
        df2['Hammer']          = (body>0) & (lower>=2*body) & (upper<=0.1*rng)
        df2['ShootingStar']    = (body>0) & (upper>=2*body) & (lower<=0.1*rng)
        df2['BullishMarubozu'] = (df2['Close']>df2['Open']) & \
                                  (df2['High'] - df2['Close'] <= buffer*rng) & \
                                  (df2['Open'] - df2['Low']   <= buffer*rng)
        df2['BearishMarubozu'] = (df2['Open']>df2['Close']) & \
                                  (df2['High'] - df2['Open'] <= buffer*rng) & \
                                  (df2['Close'] - df2['Low']  <= buffer*rng)

        be, se = [ [False] for _ in range(2) ]
        for i in range(1, len(df2)):
            p, c = df2.iloc[i-1], df2.iloc[i]
            be.append((p['Open']>p['Close']) and (c['Open']<c['Close']) and
                      (c['Open']<=p['Close']) and (c['Close']>=p['Open']))
            se.append((p['Open']<p['Close']) and (c['Open']>c['Close']) and
                      (c['Open']>=p['Close']) and (c['Close']<=p['Open']))
        df2['BullishEngulfing'], df2['BearishEngulfing'] = be, se

        ms, es = [False,False], [False,False]
        for i in range(2, len(df2)):
            f, s, t = df2.iloc[i-2], df2.iloc[i-1], df2.iloc[i]
            ms.append((f['Open']>f['Close']) and
                      abs(s['Open']-s['Close']) <= 0.3*(s['High']-s['Low']) and
                      (t['Open']<t['Close']) and
                      (t['Close']>(f['Open']+f['Close'])/2))
            es.append((f['Open']<f['Close']) and
                      abs(s['Open']-s['Close']) <= 0.3*(s['High']-s['Low']) and
                      (t['Open']>t['Close']) and
                      (t['Close']<(f['Open']+f['Close'])/2))
        df2['MorningStar'], df2['EveningStar'] = ms, es

        pl, dpc = [False], [False]
        for i in range(1, len(df2)):
            p, c = df2.iloc[i-1], df2.iloc[i]
            pl.append((p['Open']>p['Close']) and (c['Open']<c['Close']) and
                      (c['Open']<p['Low']) and (c['Close']>(p['Open']+p['Close'])/2) and
                      (c['Close']<p['Open']))
            dpc.append((p['Open']<p['Close']) and (c['Open']>c['Close']) and
                       (c['Open']>p['High']) and (c['Close']<(p['Open']+p['Close'])/2) and
                       (c['Close']>p['Open']))
        df2['PiercingLine'], df2['DarkPoolCover'] = pl, dpc

        df2['SpinningTop'] = (body>0) & (upper>=body) & (lower>=body) & (body<=0.3*rng)

        for pat in self.bullish_patterns + self.bearish_patterns + self.indecision_patterns:
            if pat not in df2.columns:
                df2[pat] = False

        cols = ['Open','High','Low','Close','Volume'] + \
               self.bullish_patterns + self.bearish_patterns + self.indecision_patterns
        return df2[[c for c in cols if c in df2.columns]]

    def detect_patterns(self, df: pd.DataFrame):
        if len(df) < 4:
            return {}
        df2 = df.copy()
        if isinstance(df2.index, pd.DatetimeIndex):
            df2 = df2.reset_index().rename(columns={'index':'Timestamp'})
        pats = self.detect_all(df2)
        for idx in range(len(pats)-1, -1, -1):
            for pat in (self.bullish_patterns + self.bearish_patterns + self.indecision_patterns):
                if pats.at[idx, pat]:
                    ts = pats.at[idx, 'Timestamp'] if 'Timestamp' in pats else pats.index[idx]
                    if isinstance(ts, (pd.Timestamp, dt.datetime)):
                        ts = ts.isoformat()
                    return {
                        pat: {
                            'index': idx,
                            'timestamp': ts,
                            'candlestick': {
                                'Open':  float(pats.at[idx,'Open']),
                                'High':  float(pats.at[idx,'High']),
                                'Low':   float(pats.at[idx,'Low']),
                                'Close': float(pats.at[idx,'Close'])
                            }
                        }
                    }
        return {}

# =========================
# AI CPR MODEL INTEGRATION
# =========================

# --- Optional: safe fallback if your project already defines this elsewhere ---
if 'convert_to_serializable' not in globals():
    def convert_to_serializable(x):
        try:
            if x is None:
                return None
            if hasattr(x, 'item'):  # numpy scalar
                return x.item()
            return float(x)
        except Exception:
            try:
                return str(x)
            except Exception:
                return None

AI_MIN_CONF = 0.55   # confidence gate (tune as you like)
AI_GATE_TRADES = True  # set False to only annotate without gating

def _ai_dir_from_label(lbl: Optional[str]):
# def _ai_dir_from_label(lbl: str | None):
    """
    Map flexible class names to directional intent:
      +1 = bullish/trend up/breakout up
      -1 = bearish/trend down/breakdown
       0 = range/sideways/neutral
     None = unknown class
    """
    if not lbl:
        return None
    s = str(lbl).lower()
    if any(k in s for k in ["bull", "up", "breakout_up", "long", "trend_up"]):
        return +1
    if any(k in s for k in ["bear", "down", "breakdown", "short", "trend_down"]):
        return -1
    if any(k in s for k in ["range", "sideways", "neutral", "meanrevert"]):
        return 0
    return None


# Add these BEFORE analyze_cpr_strategy function (around line 640)

def detect_candle_patterns(ohlc_df):
    """
    Detect key candle patterns at pivot levels
    Returns dict with pattern flags
    """
    if ohlc_df is None or len(ohlc_df) < 3:
        return {}

    patterns = {}
    try:
        # Safe access with bounds checking
        df_len = len(ohlc_df)
        if df_len < 3:
            return {}
        latest = ohlc_df.iloc[-1] if df_len > 0 else None
        prev = ohlc_df.iloc[-2] if df_len > 1 else None
        prev2 = ohlc_df.iloc[-3] if df_len > 2 else None

        if latest is None or prev is None or prev2 is None:
            return {}

        # Big Bull Take Out: Large bullish candle breaking resistance
        bull_body = latest['Close'] - latest['Open']
        patterns['big_bull_takeout'] = (
                bull_body > 0 and
                bull_body > (latest['High'] - latest['Low']) * 0.7 and
                latest['Close'] > prev['High']
        )

        # Big Bear Take Out: Large bearish candle breaking support
        bear_body = latest['Open'] - latest['Close']
        patterns['big_bear_takeout'] = (
                bear_body > 0 and
                bear_body > (latest['High'] - latest['Low']) * 0.7 and
                latest['Close'] < prev['Low']
        )

        # Fake Bull: Bullish candle rejected (next candle closes below)
        patterns['fake_bull'] = (
                prev['Close'] > prev['Open'] and
                latest['Close'] < prev['Close']
        )

        # Fake Bear: Bearish candle rejected (next candle closes above)
        patterns['fake_bear'] = (
                prev['Close'] < prev['Open'] and
                latest['Close'] > prev['Close']
        )

        # Bull Retracement: Pullback in uptrend (higher lows)
        patterns['bull_retracement'] = (
                latest['Close'] > latest['Open'] and
                latest['Low'] > prev2['Low'] and
                prev['Low'] > prev2['Low']
        )

        # Bear Retracement: Pullback in downtrend (lower highs)
        patterns['bear_retracement'] = (
                latest['Close'] < latest['Open'] and
                latest['High'] < prev2['High'] and
                prev['High'] < prev2['High']
        )

    except Exception as e:
        logger.error(f"[CANDLE-PATTERNS] Error: {e}")

    return patterns


def check_pivot_level_interactions(ohlc_df, pivot_data):
    """
    Check if price is interacting with key pivot levels
    Returns dict with interaction flags
    """
    if ohlc_df is None or len(ohlc_df) < 2:
        return {}

    interactions = {}
    try:
        latest = ohlc_df.iloc[-1]
        tolerance = 0.002  # 0.2% tolerance

        # Get pivot levels
        tc = pivot_data.get("TC", 0)
        bc = pivot_data.get("BC", 0)
        r1 = pivot_data.get("R1", 0)
        r2 = pivot_data.get("R2", 0)
        s1 = pivot_data.get("S1", 0)
        s2 = pivot_data.get("S2", 0)
        pdh = pivot_data.get("High", 0)
        pdl = pivot_data.get("Low", 0)

        # Check interactions (within tolerance)
        interactions['at_tc'] = tc and abs(latest['Close'] - tc) / tc < tolerance
        interactions['at_bc'] = bc and abs(latest['Close'] - bc) / bc < tolerance
        interactions['at_r1'] = r1 and abs(latest['Close'] - r1) / r1 < tolerance
        interactions['at_r2'] = r2 and abs(latest['Close'] - r2) / r2 < tolerance
        interactions['at_s1'] = s1 and abs(latest['Close'] - s1) / s1 < tolerance
        interactions['at_s2'] = s2 and abs(latest['Close'] - s2) / s2 < tolerance
        interactions['at_pdh'] = pdh and abs(latest['Close'] - pdh) / pdh < tolerance
        interactions['at_pdl'] = pdl and abs(latest['Close'] - pdl) / pdl < tolerance

        # Check if testing levels (wicks)
        interactions['tested_tc'] = tc and latest['High'] >= tc * 0.998 and latest['Low'] <= tc * 1.002
        interactions['tested_bc'] = bc and latest['High'] >= bc * 0.998 and latest['Low'] <= bc * 1.002
        interactions['tested_pdh'] = pdh and latest['High'] >= pdh * 0.998
        interactions['tested_pdl'] = pdl and latest['Low'] <= pdl * 1.002

    except Exception as e:
        logger.error(f"[PIVOT-INTERACTIONS] Error: {e}")

    return interactions

# ==============================
# UPDATED CPR ANALYSIS (WITH AI)
# ==============================
def analyze_cpr_strategy(indicators, pivot_data, ai_predictor, ohlc_df=None):
    """
    Analyze market using Enhanced CPR with Key Price Action Levels and Candle Patterns
    """
    # ✅ Check for minimal required indicators before proceeding
    has_minimal_data = all([
        indicators.get("ema_5"),
        indicators.get("ema_21"),
        indicators.get("close")
    ])

    if not has_minimal_data:
        return {"error": f"Insufficient indicators for CPR analysis: ema_5={indicators.get('ema_5')}, ema_21={indicators.get('ema_21')}, close={indicators.get('close')}",
                "trade_strategy": "None"}

    ema_21 = convert_to_serializable(indicators.get("ema_21", 0))
    ema_5 = convert_to_serializable(indicators.get("ema_5", 0))
    ema_20 = convert_to_serializable(indicators.get("ema_20", 0))
    ema_50 = convert_to_serializable(indicators.get("ema_50", 0))
    ema_200 = convert_to_serializable(indicators.get("ema_200", 0))
    st21Trend = convert_to_serializable(indicators.get("st21Trend", 0))
    adx = convert_to_serializable(indicators.get("adx", 0))
    close = convert_to_serializable(indicators.get("close", 0))

    # CPR/Pivot levels
    tc = convert_to_serializable(pivot_data.get("TC", 0))
    bc = convert_to_serializable(pivot_data.get("BC", 0))

    r1 = convert_to_serializable(pivot_data.get("R1", 0))
    r2 = convert_to_serializable(pivot_data.get("R2", 0))
    r3 = convert_to_serializable(pivot_data.get("R3", 0))
    s1 = convert_to_serializable(pivot_data.get("S1", 0))
    s2 = convert_to_serializable(pivot_data.get("S2", 0))
    s3 = convert_to_serializable(pivot_data.get("S3", 0))

    # Check if essential data is available and valid
    if not (tc and bc and close and tc > 0 and bc > 0 and close > 0):
        return {"error": f"Invalid CPR data: TC={tc or 0}, BC={bc or 0}, Close={close or 0}",
                "trade_strategy": "None"}

    # Key Price Action Levels
    pdh = convert_to_serializable(pivot_data.get("High", 0))
    pdl = convert_to_serializable(pivot_data.get("Low", 0))
    pwh = convert_to_serializable(pivot_data.get("PWH", 0))
    pwl = convert_to_serializable(pivot_data.get("PWL", 0))
    pmh = convert_to_serializable(pivot_data.get("PMH", 0))
    pml = convert_to_serializable(pivot_data.get("PML", 0))
    wh_52 = convert_to_serializable(pivot_data.get("52WH", 0))
    wl_52 = convert_to_serializable(pivot_data.get("52WL", 0))

    # Previous day CPR for comparison (if available)
    prev_tc = convert_to_serializable(pivot_data.get("prev_TC"))
    prev_bc = convert_to_serializable(pivot_data.get("prev_BC"))

    if not all([tc, bc, close]):
        return {"error": "Invalid CPR or price data"}

    # Key Price Action View Formulation (Market Bias)
    key_price_action_view = "NEUTRAL"
    position_sizing = "CONSERVATIVE"

    # AGGRESSIVE BULLISH MOMENTUM: Break above key levels with follow-through
    if (close > pdh * 1.002 and adx > 35 and ema_5 > ema_21):
        key_price_action_view = "AGGRESSIVE_BULLISH"
        position_sizing = "AGGRESSIVE"
    elif (close > pmh * 1.002 and adx > 30):  # Break above monthly high
        key_price_action_view = "AGGRESSIVE_BULLISH"
        position_sizing = "AGGRESSIVE"
    elif (close > wh_52 * 1.002 and adx > 25):  # Break above 52-week high
        key_price_action_view = "AGGRESSIVE_BULLISH"
        position_sizing = "AGGRESSIVE"

    # AGGRESSIVE BEARISH REVERSAL: Rejection at resistance or break below support
    elif (close < pdl * 0.998 and adx > 35 and ema_5 < ema_21):
        key_price_action_view = "AGGRESSIVE_BEARISH"
        position_sizing = "AGGRESSIVE"
    elif (close < pml * 0.998 and adx > 30):  # Break below monthly low
        key_price_action_view = "AGGRESSIVE_BEARISH"
        position_sizing = "AGGRESSIVE"
    elif (close < wl_52 * 0.998 and adx > 25):  # Break below 52-week low
        key_price_action_view = "AGGRESSIVE_BEARISH"
        position_sizing = "AGGRESSIVE"

    # DEFENSIVE RETRACEMENT: Break then reverse (caution mode)
    elif (close > pdh * 1.001 and close < pdh * 1.005 and adx < 30):
        key_price_action_view = "DEFENSIVE_RETRACEMENT"
        position_sizing = "DEFENSIVE"
    elif (close < pdl * 0.999 and close > pdl * 0.995 and adx < 30):
        key_price_action_view = "DEFENSIVE_RETRACEMENT"
        position_sizing = "DEFENSIVE"

    # SWING REVERSAL: Invalidating previous view
    elif (pwh and close < pwh * 0.998):  # Break below previous week high (long invalidation)
        key_price_action_view = "SWING_REVERSAL_SHORT"
        position_sizing = "AGGRESSIVE"
    elif (pwl and close > pwl * 1.002):  # Break above previous week low (short invalidation)
        key_price_action_view = "SWING_REVERSAL_LONG"
        position_sizing = "AGGRESSIVE"

    # CPR Width Analysis (Trend Bias)
    cpr_width = tc - bc if tc and bc else 0
    cpr_trend_bias = "NEUTRAL"
    if cpr_width > 0:
        # Compare with typical width (simplified - could be enhanced with historical avg)
        if cpr_width > (close * 0.02):  # > 2% of price
            cpr_trend_bias = "WIDE"  # Sideways/Range bound
        else:
            cpr_trend_bias = "NARROW"  # Trending day expected

    # CPR Position Analysis
    cpr_position = "NEUTRAL"
    if close > tc:
        cpr_position = "ABOVE_TC"  # Bullish bias
    elif close < bc:
        cpr_position = "BELOW_BC"  # Bearish bias
    elif bc <= close <= tc:
        cpr_position = "IN_CPR"  # Neutral/Range bound

    # CPR vs Previous Day Analysis
    cpr_vs_prev = "NEUTRAL"
    if prev_tc and prev_bc:
        if tc > prev_tc and bc > prev_bc:  # Shifted up
            cpr_vs_prev = "SHIFTED_UP"
        elif tc < prev_tc and bc < prev_bc:  # Shifted down
            cpr_vs_prev = "SHIFTED_DOWN"
        elif (prev_bc <= tc <= prev_tc and prev_bc <= bc <= prev_tc):  # Inside previous
            cpr_vs_prev = "INSIDE_PREV"
        elif (tc > prev_tc or bc < prev_bc):  # Outside previous
            cpr_vs_prev = "OUTSIDE_PREV"

    # MA Trend Analysis (20>50/50>20 as per PDF)
    ma_trend = "NEUTRAL"
    if ema_20 and ema_50:
        if ema_20 > ema_50:
            ma_trend = "BULLISH"
        elif ema_50 > ema_20:
            ma_trend = "BEARISH"

    # 200MA as support/resistance (as per PDF)
    ma_200_signal = "NEUTRAL"
    if ema_200:
        if close > ema_200 and ema_200 > 0:
            ma_200_signal = "ABOVE_200MA"
        elif close < ema_200 and ema_200 > 0:
            ma_200_signal = "BELOW_200MA"

    # AI CPR inference
    ai_label, ai_conf, ai_dist, _ = ai_predictor.predict(indicators, pivot_data, _build_ai_cpr_features)
    ai_dir = _ai_dir_from_label(ai_label)

    # Candle pattern detection at pivot levels
    candle_patterns = {}
    pivot_interactions = {}
    if ohlc_df is not None and not ohlc_df.empty:
        candle_patterns = detect_candle_patterns(ohlc_df)
        pivot_interactions = check_pivot_level_interactions(ohlc_df, pivot_data)

    # Reversal detection at R2/R3 for exit management
    reversal_at_r2_r3 = False
    if ohlc_df is not None and len(ohlc_df) >= 3:
        latest = ohlc_df.iloc[-1]
        prev = ohlc_df.iloc[-2]
        # Bearish reversal at resistance (R2/R3)
        if (latest['Close'] < latest['Open'] and  # Bear candle
                (r2 or r3) and
                latest['High'] >= (r2 or r3) * 0.995 and  # Tested resistance
                latest['Close'] < (r2 or r3) * 0.998):  # Rejected
            reversal_at_r2_r3 = True

    trade_strategy = "None"
    reason = ""

    # AGGRESSIVE LONG ENTRY (Confirmed Trend) - Must be above all key levels
    if (key_price_action_view == "AGGRESSIVE_BULLISH" and
            close > tc and close > pdh and close > r1 and close > r2 and
            adx > 30 and ema_5 > ema_21):
        trade_strategy = "Buy"
        reason = f"AGGRESSIVE LONG: Above CPR,PDH,R1,R2 | View:{key_price_action_view} | MA:{ma_trend}"

    # AGGRESSIVE LONG ENTRY (Confirmed Support/Pullback Reversal)
    elif (key_price_action_view in ["AGGRESSIVE_BULLISH", "DEFENSIVE_RETRACEMENT"] and
          close > pmh * 0.998 and  # At monthly support
          close > tc and close > r1 and  # Breaking CPR and R1
          adx > 25 and ema_5 > ema_21 and
          candle_patterns.get('bull_retracement')):
        trade_strategy = "Buy"
        reason = f"AGGRESSIVE LONG: PMH support, broke CPR/R1 | View:{key_price_action_view} | Pattern: Bull Retracement"

    # AGGRESSIVE SHORT ENTRY (Symmetrical to long - using S1/S2)
    elif (key_price_action_view == "AGGRESSIVE_BEARISH" and
          close < bc and close < pdl and close < s1 and close < s2 and
          adx > 30 and ema_5 < ema_21):
        trade_strategy = "Sell"
        reason = f"AGGRESSIVE SHORT: Below CPR,PDL,S1,S2 | View:{key_price_action_view} | MA:{ma_trend}"

    # AGGRESSIVE SHORT ENTRY (Support break)
    elif (key_price_action_view in ["AGGRESSIVE_BEARISH", "DEFENSIVE_RETRACEMENT"] and
          close < pml * 1.002 and  # At monthly resistance
          close < bc and close < s1 and  # Breaking CPR and S1
          adx > 25 and ema_5 < ema_21 and
          candle_patterns.get('bear_retracement')):
        trade_strategy = "Sell"
        reason = f"AGGRESSIVE SHORT: PML resistance, broke CPR/S1 | View:{key_price_action_view} | Pattern: Bear Retracement"

    # Enhanced strategy selection with Key Price Action + Candle Patterns + MA Analysis + CPR Rules
    elif close > pdh * 1.002:  # Break above PDH
        if adx and adx > 35 and close > tc and close > r1 and ema_5 > ema_21:
            # Check for bullish candle patterns at pivot levels
            bullish_signals = []
            if candle_patterns.get('big_bull_takeout'):
                bullish_signals.append("Big Bull Take Out")
            if candle_patterns.get('fake_bear'):
                bullish_signals.append("Fake Bear")
            if candle_patterns.get('bull_retracement'):
                bullish_signals.append("Bull Retracement")

            # Add MA trend confirmation
            ma_confirmed = ma_trend == "BULLISH" or ma_200_signal == "ABOVE_200MA"

            if bullish_signals:
                trade_strategy = "Buy"
                reason = f"AGGRESSIVE: Break above PDH, above TC/R1, EMA5>21, ADX>{adx:.0f} | MA:{ma_trend}/{ma_200_signal} | Candles: {', '.join(bullish_signals)}"
            elif ma_confirmed:
                trade_strategy = "Buy"
                reason = f"AGGRESSIVE: Break above PDH, above TC/R1, EMA5>21, ADX>{adx:.0f} | MA:{ma_trend}/{ma_200_signal}"
            else:
                trade_strategy = "Buy"
                reason = f"AGGRESSIVE: Break above PDH, above TC/R1, EMA5>21, ADX>{adx:.0f}"
    elif close < pdl * 0.998:  # Break below PDL
        if adx and adx > 35 and close < bc and close < s1 and ema_5 < ema_21:
            # Check for bearish candle patterns at pivot levels
            bearish_signals = []
            if candle_patterns.get('big_bear_takeout'):
                bearish_signals.append("Big Bear Take Out")
            if candle_patterns.get('fake_bull'):
                bearish_signals.append("Fake Bull")
            if candle_patterns.get('bear_retracement'):
                bearish_signals.append("Bear Retracement")

            # Add MA trend confirmation
            ma_confirmed = ma_trend == "BEARISH" or ma_200_signal == "BELOW_200MA"

            if bearish_signals:
                trade_strategy = "Sell"
                reason = f"AGGRESSIVE: Break below PDL, below BC/S1, EMA5<21, ADX>{adx:.0f} | MA:{ma_trend}/{ma_200_signal} | Candles: {', '.join(bearish_signals)}"
            elif ma_confirmed:
                trade_strategy = "Sell"
                reason = f"AGGRESSIVE: Break below PDL, below BC/S1, EMA5<21, ADX>{adx:.0f} | MA:{ma_trend}/{ma_200_signal}"
            else:
                trade_strategy = "Sell"
                reason = f"AGGRESSIVE: Break below PDL, below BC/S1, EMA5<21, ADX>{adx:.0f}"
    elif close > tc and close > pdh:
        key_price_action_view = "BULLISH_MOMENTUM"
        if adx and ema_5 > ema_21 and ma_trend == "BULLISH":
            trade_strategy = "Buy"
            reason = f"MOMENTUM: Above TC & PDH, EMA5>21, ADX>{adx:.0f} | MA:{ma_trend}"
    elif close < bc and close < pdl:
        key_price_action_view = "BEARISH_MOMENTUM"
        if adx and adx > 20 and ema_5 < ema_21 and ma_trend == "BEARISH":
            trade_strategy = "Sell"
            reason = f"MOMENTUM: Below BC & PDL, EMA5<21, ADX>{adx:.0f} | MA:{ma_trend}"

    # NEW: CPR Breakout Rules (Page 37)
    elif cpr_position == "ABOVE_TC" and cpr_trend_bias == "NARROW" and ma_trend == "BULLISH":
        # Bullish breakout setup
        if adx > 25 and ema_5 > ema_21:
            trade_strategy = "Buy"
            reason = f"CPR BREAKOUT: Above TC, narrow width, EMA5>21, ADX>{adx:.0f} | MA:{ma_trend}"
    elif cpr_position == "BELOW_BC" and cpr_trend_bias == "NARROW" and ma_trend == "BEARISH":
        # Bearish breakout setup
        if adx > 25 and ema_5 < ema_21:
            trade_strategy = "Sell"
            reason = f"CPR BREAKOUT: Below BC, narrow width, EMA5<21, ADX>{adx:.0f} | MA:{ma_trend}"

    # NEW: CPR Retest Rules
    elif cpr_position == "ABOVE_TC" and close < tc * 1.001:  # Retesting TC from above (support)
        if adx > 20 and ma_trend == "BULLISH" and candle_patterns.get('bull_retracement'):
            trade_strategy = "Buy"
            reason = f"CPR SUPPORT: Retest TC, bullish MA, ADX>{adx:.0f} | Pattern: Bull Retracement"
    elif cpr_position == "BELOW_BC" and close > bc * 0.999:  # Retesting BC from below (resistance)
        if adx > 20 and ma_trend == "BEARISH" and candle_patterns.get('bear_retracement'):
            trade_strategy = "Sell"
            reason = f"CPR RESISTANCE: Retest BC, bearish MA, ADX>{adx:.0f} | Pattern: Bear Retracement"

    # NEW: Confluence Setups (High Probability)
    elif (cpr_position in ["ABOVE_TC", "BELOW_BC"] and
          cpr_vs_prev in ["SHIFTED_UP", "SHIFTED_DOWN", "OUTSIDE_PREV"] and
          cpr_trend_bias == "NARROW" and
          ma_trend != "NEUTRAL"):
        if cpr_position == "ABOVE_TC" and ma_trend == "BULLISH":
            trade_strategy = "Buy"
            reason = f"CONFLUENCE: CPR breakout, {cpr_vs_prev}, narrow width, bullish MA"
        elif cpr_position == "BELOW_BC" and ma_trend == "BEARISH":
            trade_strategy = "Sell"
            reason = f"CONFLUENCE: CPR breakout, {cpr_vs_prev}, narrow width, bearish MA"

    # EXIT MANAGEMENT: Reversal at R2/R3 for aggressive long positions
    elif reversal_at_r2_r3 and key_price_action_view == "AGGRESSIVE_BULLISH":
        trade_strategy = "Exit"
        reason = f"EXIT: Reversal at R2/R3 resistance | View:{key_price_action_view}"

    # AI veto/assist
    ai_filter_pass = True
    if ai_dir is not None and ai_conf is not None:
        if trade_strategy.startswith("Buy") and ai_dir < 0 and ai_conf >= AI_MIN_CONF:
            ai_filter_pass = False
            reason += f" | AI disagrees (label={ai_label}, conf={round(ai_conf, 2)})"
        elif trade_strategy.startswith("Sell") and ai_dir > 0 and ai_conf >= AI_MIN_CONF:
            ai_filter_pass = False
            reason += f" | AI disagrees (label={ai_label}, conf={round(ai_conf, 2)})"
        elif trade_strategy == "Exit" and ai_dir < 0 and ai_conf >= AI_MIN_CONF:
            # AI confirms exit
            reason += f" | AI confirms exit (label={ai_label}, conf={round(ai_conf, 2)})"
        else:
            reason += f" | AI:{ai_label}({round(ai_conf, 2) if ai_conf is not None else 'NA'})"

    return {
        "trade_strategy": trade_strategy,
        "reason": reason,
        "ai_cpr_label": ai_label,
        "ai_confidence": ai_conf,
        "ai_distribution": ai_dist,
        "ai_filter_pass": ai_filter_pass,
        "key_price_action_view": key_price_action_view,
        "position_sizing": position_sizing,
        "candle_patterns": candle_patterns,
        "pivot_interactions": pivot_interactions,
        "reversal_at_r2_r3": reversal_at_r2_r3,
        "ma_trend": ma_trend,
        "ma_200_signal": ma_200_signal,
        "cpr_trend_bias": cpr_trend_bias,
        "cpr_position": cpr_position,
        "cpr_vs_prev": cpr_vs_prev,
        "cpr_width": cpr_width,
        # 🔥 ADD THIS LINE:
        "cpr_levels": pivot_data  # ← Pass through the original CPR levels!
    }

# ───────────────────────────────────────────────────────────────────────────────
# IndicatorCalculator.calculate_indicators (full replacement)
# ───────────────────────────────────────────────────────────────────────────────
class IndicatorCalculator:
    def __init__(self, bot):
        self.bot = bot

    def calculate_pivot_points(self, df_day):
        if df_day.empty:
            self.bot.log_message("Pivot: empty df", False)
            return {}
        for c in ["High", "Low", "Close"]:
            if c not in df_day:
                self.bot.log_message(f"Pivot: missing {c}", False)
                return {}

        # --- Today's Pivots (from Previous Day's Data) ---
        prev_day = df_day.iloc[-2]
        high  = round(prev_day["High"],  2)
        low   = round(prev_day["Low"],   2)
        close = round(prev_day["Close"], 2)

        PP = round((high + low + close) / 3, 2)
        BC = round((high + low) / 2, 2)
        TC = round((PP - BC) + PP, 2)
        if TC < BC: TC, BC = BC, TC # Ensure TC is always above BC

        # --- Virgin CPR Check (for today's pivots) ---
        # A CPR is virgin if the *previous day's* price action did not touch it.
        virgin_cpr = False
        if prev_day["High"] < BC or prev_day["Low"] > TC:
            virgin_cpr = True

        return {
            # CPR Levels (existing)
            "PP": PP, "TC": TC, "BC": BC,
            "High": high, "Low": low, "Close": close,
            "virgin_cpr": virgin_cpr,

            # Standard Pivot Levels (existing)
            "R1": round(2 * PP - low, 2),
            "S1": round(2 * PP - high, 2),
            "R2": round(PP + (high - low), 2),
            "S2": round(PP - (high - low), 2),
            "R3": round(high + 2 * (PP - low), 2),
            "S3": round(low - 2 * (high - PP), 2),

            # ✅ ADD THESE NEW FIELDS:
            "PWH": round(df_day['High'].tail(5).max(), 2) if len(df_day) >= 5 else high,  # Previous week high
            "PWL": round(df_day['Low'].tail(5).min(), 2) if len(df_day) >= 5 else low,  # Previous week low
            "PMH": round(df_day['High'].tail(20).max(), 2) if len(df_day) >= 20 else high,  # Previous month high
            "PML": round(df_day['Low'].tail(20).min(), 2) if len(df_day) >= 20 else low,  # Previous month low
            "52WH": round(df_day['High'].tail(252).max(), 2) if len(df_day) >= 252 else high,  # 52-week high
            "52WL": round(df_day['Low'].tail(252).min(), 2) if len(df_day) >= 252 else low,  # 52-week low
            "prev_TC": round(df_day['TC'].iloc[-2], 2) if 'TC' in df_day.columns and len(df_day) >= 2 else None,
            "prev_BC": round(df_day['BC'].iloc[-2], 2) if 'BC' in df_day.columns and len(df_day) >= 2 else None,
        }

    def calculate_support_resistance(self, df, period=20):
        df["Support"]    = df["Low"].rolling(period).min()
        df["Resistance"] = df["High"].rolling(period).max()
        return df

    def calculate_indicators(self, symbol, timeframe, pivot_data=None, ohlc_df=None):
        # 1) Fetch & validate OHLC. Use provided DataFrame if available (for backtesting).
        if ohlc_df is not None and not ohlc_df.empty:
            ohlc = ohlc_df.copy()
            self.bot.log_message("IndicatorCalc: Using pre-fetched OHLC for backtesting.", True)
        else:
            ohlc = self.bot.fetch_ohlc(symbol, timeframe, 60)

        # --- ADDED: Robustness check for backtesting ---
        # Ensure there's enough data for lookbacks (e.g., iloc[-2], rolling windows)
        # A minimum of 30 is a safe starting point for most indicators used.
        if len(ohlc) < 30:
            self.bot.log_message(f"IndicatorCalc: Not enough data ({len(ohlc)} bars) for full calculation. Need at least 30.", True)
            return {"error": f"Not enough historical data ({len(ohlc)} bars)."}
        # --- END ADD ---

        if ohlc.empty:
            self.bot.log_message(
                f"IndicatorCalc: No valid OHLC data for {symbol} after filtering.",
                False
            )
            return {"error": "No data available after filtering."}

        # 2) Baseline transforms
        ohlc.index = pd.to_datetime(ohlc.index, utc=True)
        ohlc = vwap(ohlc)

        # ATR with safe fallback
        ohlc["ATR"] = atr(ohlc, 14)
        atr_series = pd.to_numeric(ohlc["ATR"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if atr_series.isna().all():
            atr_series = (ohlc["High"] - ohlc["Low"]).rolling(14, min_periods=1).mean()
        ohlc["ATR"] = atr_series.ffill().bfill()
        ohlc["atr"] = ohlc["ATR"]

        # Bands & EMAs
        ohlc = bollinger_bands(ohlc)
        for span in (5, 9, 21, 50, 200):
            ohlc = ema(ohlc, span)

        ohlc["EMA20"] = ohlc["Close"].ewm(span=20, adjust=False).mean()
        # --- MACD (12/26/9) + histogram (+ prev) + stds (robust color) ---
        ema_fast    = ohlc["Close"].ewm(span=12, adjust=False).mean()
        ema_slow    = ohlc["Close"].ewm(span=26, adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist   = macd_line - macd_signal
        ohlc["MACD_hist"] = macd_hist

        # statistics we later use
        macd_spread = macd_line - macd_signal
        hist_std    = macd_hist.rolling(50, min_periods=10).std()
        spread_std  = macd_spread.rolling(50, min_periods=10).std()

        hist_now  = float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else None
        hist_prev = float(macd_hist.iloc[-2]) if pd.notna(macd_hist.iloc[-2]) else None
        line_now  = float(macd_line.iloc[-1]) if pd.notna(macd_line.iloc[-1]) else None
        sig_now   = float(macd_signal.iloc[-1]) if pd.notna(macd_signal.iloc[-1]) else None

        # ===== robust color classification =====
        hist_std_last = float(hist_std.iloc[-1]) if pd.notna(hist_std.iloc[-1]) else None
        eps = 1e-7
        if hist_std_last is not None and hist_std_last > 0:
            eps = max(eps, 0.05 * hist_std_last)  # 5% of σ50

        macd_color = "Neutral"
        if hist_now is not None and hist_prev is not None:
            d = hist_now - hist_prev
            if   hist_now >=  eps and d >  +eps: macd_color = "Dark Green"
            elif hist_now >=  eps and d <= +eps: macd_color = "Light Green"
            elif hist_now <= -eps and d <  -eps: macd_color = "Dark Red"
            elif hist_now <= -eps and d >= -eps: macd_color = "Light Red"
            # else Neutral (|hist_now| < eps)

        # ===== 'flat' detector (line≈signal AND tiny histogram) =====
        spread_std_last = float(spread_std.iloc[-1]) if pd.notna(spread_std.iloc[-1]) else None
        macd_is_flat = False
        if (line_now is not None and sig_now is not None and
            spread_std_last is not None and spread_std_last > 0 and
            hist_std_last is not None and hist_std_last > 0 and
            hist_now is not None):
            near_line = abs(line_now - sig_now) <= 0.10 * spread_std_last
            tiny_hist = abs(hist_now)           <= 0.20 * hist_std_last
            macd_is_flat = near_line and tiny_hist

        # ADX(+DI/−DI)
        high, low, close = ohlc["High"], ohlc["Low"], ohlc["Close"]
        plus_dm  = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr_cols  = pd.concat([high - low,
                            (high - close.shift()).abs(),
                            (low - close.shift()).abs()], axis=1)
        tr    = tr_cols.max(axis=1)
        atr14 = tr.rolling(14, min_periods=14).mean()

        plus_di  = 100 * (plus_dm.rolling(14, min_periods=14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14, min_periods=14).mean() / atr14)
        dx   = 100 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di))
        adx  = dx.rolling(14, min_periods=14).mean()
        ohlc["ADX_series"] = adx
        ohlc["DI_plus"]    = plus_di
        ohlc["DI_minus"]   = minus_di

        # S/R + Volume
        ohlc["Support"]    = ohlc["Low"].rolling(20).min()
        ohlc["Resistance"] = ohlc["High"].rolling(20).max()
        ohlc["VolSMA20"]   = ohlc["Volume"].rolling(20, min_periods=1).mean()
        ohlc["volume_sma_20"] = ohlc["VolSMA20"]

        if len(ohlc) < 2:
            self.bot.log_message(
                f"IndicatorCalc: Not enough data points ({len(ohlc)}) for {symbol} to calculate indicators.",
                False
            )
            return {"error": "Not enough data for indicators."}

        latest, prev = ohlc.iloc[-1], ohlc.iloc[-2]

        # EMA50/200 summary (safe)
        try:
            ema50 = EMA50_200(self.bot.fyers_sdk_instance, symbol, timeframe, 40)
            ema_sig = ema50.get_current_signal()
            if len(ema50.df) < 50:
                ema_sig = {"signal": "NO TRADE", "trend": "Neutral", "trend_strength": 0}
        except Exception:
            ema_sig = {"signal": "NO TRADE", "trend": "Neutral", "trend_strength": 0}

        # Supertrend (21/14/7)
        st21, _, _, tr21 = supertrend(ohlc, 21, 1)
        st14, _, _, tr14 = supertrend(ohlc, 14, 2)
        st7,  _, _, tr7  = supertrend(ohlc, 7,  3)

        # Extras (unchanged)
        bull_div, bear_div = rsi_divergence(ohlc)
        patterns   = self.bot.candle_analyzer.detect_patterns(ohlc)
        adx_bundle = adx_efi_mom_trade_signal(fetchOHLC1(symbol, interval="5", duration=60), symbol)
        fib        = fibonacci_retracement(ohlc, logger=self.bot.log_message)

        # --- NEW: CPR & AI Analysis ---
        #cpr_analysis = analyze_cpr_strategy(latest.to_dict(), pivot_data or {}, self.bot.ai_predictor)
        # NEW CALL (add ohlc_df):
        cpr_analysis = analyze_cpr_strategy(
            indicators=latest.to_dict(),
            pivot_data=pivot_data or {},
            ai_predictor=self.bot.ai_predictor,
            ohlc_df=ohlc  # ✅ Pass full OHLC dataframe for candle pattern detection
        )

        return {
            "timestamp": latest.name.isoformat(),
            "close": float(latest["Close"]),
            "close_prev": float(prev["Close"]),
            "high_prev": float(prev["High"]),
            "low_prev": float(prev["Low"]),

            # EMAs (+prev)
            "ema_5": float(latest["EMA5"]), "ema_9": float(latest["EMA9"]), "ema_21": float(latest["EMA21"]),
            "ema_50": float(latest["EMA50"]), "ema_200": float(latest["EMA200"]),
            "ema_5_prev": float(prev["EMA5"]), "ema_9_prev": float(prev["EMA9"]),
            "ema_20": float(latest["EMA20"]),
            "ema_21_prev": float(prev["EMA21"]), "ema_50_prev": float(prev["EMA50"]),
            "ema_200_prev": float(prev["EMA200"]),

            # Volume (+prev, +SMA20)
            "volume": float(latest["Volume"]),
            "volume_prev": float(prev["Volume"]),
            "volume_sma_20": float(ohlc["VolSMA20"].iloc[-1]),

            # MACD (+line/signal & stds for strategy)
            "macd_line": line_now,
            "macd_signal": sig_now,
            "macd_hist": hist_now,
            "macd_hist_prev": hist_prev,
            "macd_hist_std50": (float(hist_std.iloc[-1]) if pd.notna(hist_std.iloc[-1]) else None),
            "macd_spread_std50": (float(spread_std.iloc[-1]) if pd.notna(spread_std.iloc[-1]) else None),
            "macd_color": macd_color,
            "macd_is_flat": macd_is_flat,

            # ADX (+prev)
            "adx": float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else None,
            "adx_prev": float(adx.iloc[-2]) if pd.notna(adx.iloc[-2]) else None,

            # Other indicators
            "supertrend": float(st21.iloc[-1]) if pd.notna(st21.iloc[-1]) else None,
            "BB_upper": float(latest.get("BB_upper")) if pd.notna(latest.get("BB_upper")) else None,
            "BB_lower": float(latest.get("BB_lower")) if pd.notna(latest.get("BB_lower")) else None,
            "BB_mid": float(latest.get("BB_mid")) if pd.notna(latest.get("BB_mid")) else None,
            "VWAP": float(latest["VWAP"]),
            "support": float(latest["Support"]), "resistance": float(latest["Resistance"]),
            # --- CORRECTED: Bollinger Bandwidth for volatility filter ---
            "bb_bandwidth": ((latest.get("BB_upper", 0) - latest.get("BB_lower", 0)) / latest.get("BB_mid", 1))
                            if (latest.get("BB_mid") and latest.get("BB_mid") > 0 and
                                latest.get("BB_upper") and latest.get("BB_lower"))
                            else 0.0,

            "ATR": float(latest["ATR"]),

            # EMA50/200 summary
            "ema50_200_signal": ema_sig["signal"],
            "ema50_200_trend":  ema_sig["trend"] ,

            # Ultimate MA / Super-Guppy summary & ST trend flags
            "super_guppy": UltimateMAIndicator(ohlc, {"len": 13, "len2": 34}).summarized(),
            "st21Trend": int(tr21.iloc[-1]) if not tr21.empty else 0,
            "st14Trend": int(tr14.iloc[-1]) if not tr14.empty else 0,
            "st7Trend":  int(tr7.iloc[-1])  if not tr7.empty  else 0,

            # Patterns, ADX/EFI/MOM bundle, Fib, RSI divergences
            "patterns": patterns,
            "adx_efi": {
                "signal": adx_bundle[0], "ADX": adx_bundle[1], "DI+": adx_bundle[2],
                "DI-": adx_bundle[3], "Momentum": adx_bundle[4], "EFI": adx_bundle[5]
            },
            "fib": fib,
            "rsi_div": {"bull": bull_div, "bear": bear_div},
            "cpr_analysis": cpr_analysis,
        }



class FyersService:
    def __init__(self, fyers_sdk, raw_log_path, log_fn, websocket_ltp_fn=None):
        self.sdk     = fyers_sdk
        self.raw_log = raw_log_path
        self.log     = log_fn
        self.get_websocket_ltp = websocket_ltp_fn

    def place_market_order(self, symbol, side, qty):
        self.log(f"Market {side} {symbol} qty={qty}", debug_only=True)
        ltp = self.get_websocket_ltp(symbol, timeout=5) if self.get_websocket_ltp else None

        payload = {
            "symbol": symbol,
            "qty": int(qty),
            "type": 2,
            "side": (1 if side == "BUY" else -1),
            "productType": "INTRADAY",
            "limitPrice": 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": 0,
            "takeProfit": 0
        }
        resp = call_with_rate_limit_retry(self.sdk.place_order, data=payload)

        now = dt.datetime.now(IST).isoformat()
        event = {
            "timestamp": now,
            "action":    "place_market",
            "symbol":    symbol,
            "side":      side,
            "qty":       qty,
            "ltp":       ltp,
            "payload":   payload,
            "response":  resp
        }
        save_order_data_event(event, self.raw_log)
        return resp, None

    def exit_all_positions_for_symbol(self, symbol: str):
        """
        Close ONLY the position matching `symbol` using the positions DELETE API.
        - Finds the matching position id from GET /positions (netPositions).
        - Sends {"id": <position_id>} to DELETE /positions.
        - Does NOT send `symbol` in the delete payload (the API ignores/doesn't expect it).
        """
        self.log(f"Exit (single symbol) request for {symbol}", debug_only=False)

        # 1) Find the position id for this symbol
        pos_id = None
        positions = call_with_rate_limit_retry(self.sdk.positions) or {}
        net_positions = positions.get("netPositions") or []
        try:
            for p in net_positions:
                candidates = [p.get("symbol"), p.get("symbolName"), p.get("tradingSymbol"), p.get("segmentSymbol")]
                p_symbol = next((s for s in candidates if isinstance(s, str) and s.strip()), None)
                if (p_symbol or "").strip() != symbol:
                    continue
                raw_qty = (p.get("netQty") or p.get("qty") or p.get("net_quantity") or p.get("quantity"))
                try:
                    net_qty = float(str(raw_qty).strip())
                except Exception:
                    net_qty = 0.0
                if abs(net_qty) < 1e-6:
                    continue
                pos_id = p.get("id") or p.get("positionId") or p.get("posId")
                if pos_id:
                    break
        except Exception as e:
            self.log(f"[EXIT] Failed to parse positions: {e}", debug_only=False)

        if not pos_id:
            self.log(f"[EXIT] No open position id found for {symbol}.", debug_only=False)
            resp = {"s": "error", "code": -66, "message": "No open position for symbol"}
            event = {
                "timestamp": dt.datetime.now(IST).isoformat(),
                "action":    "exit_symbol",
                "symbol":    symbol,
                "ltp":       self.get_websocket_ltp(symbol, timeout=5) if self.get_websocket_ltp else None,
                "payload":   {"id": None, "note": "no matching open position id"},
                "response":  resp
            }
            save_order_data_event(event, self.raw_log)
            return resp, None

        payload = {"id": pos_id}
        ltp = self.get_websocket_ltp(symbol, timeout=5) if self.get_websocket_ltp else None
        resp = call_with_rate_limit_retry(self.sdk.exit_positions, data=payload)

        event = {
            "timestamp": dt.datetime.now(IST).isoformat(),
            "action":    "exit_symbol",
            "symbol":    symbol,
            "ltp":       ltp,
            "payload":   payload,
            "response":  resp
        }
        save_order_data_event(event, self.raw_log)
        return resp, None



def save_order_data_event(event, path):
    lst = robust_load_json(path, print, default=[])
    if not isinstance(lst, list):
        lst = []
    lst.append(convert_to_serializable(event))
    robust_save_json(lst, path, print, debug_only=True)


# ==============================
# ENHANCED OrderManager CLASS WITH AI CPR
# ==============================
class OrderManager:
    # -----------------------------
    # CONFIGURATION CONSTANTS
    # -----------------------------
    # Paper trading mode
    PAPER_TRADING_MODE = False

    # AI CPR Configuration
    AI_CPR_ENABLED = True
    AI_MIN_CONF = 0.55  # Minimum confidence for AI signal acceptance
    AI_SOLO_MIN_CONF = 0.75  # Higher threshold for AI-only entries
    AI_SOLO_MAX_OPPOSITION = 2  # Max opposing signals for AI solo entry
    AI_GATE_TRADES = True  # Require AI confirmation for entries

    # Entry Thresholds
    MIN_VOTES_STRONG = 3  # Strong confluence (50% agreement)
    MIN_VOTES_MEDIUM = 2  # Medium confluence
    MEDIUM_CONF_MIN_SCORE = 1.5  # Minimum score for 2-vote entry without AI
    MEDIUM_CONF_MIN_AI = 0.65  # Minimum AI confidence for 2-vote entry with AI
    MEDIUM_CONF_MIN_SCORE_WITH_AI = 1.3  # Lower score if AI involved

    # Risk Management
    STRONG_TREND_ATR_GAP = 1.5
    CONSOL_TOL_NARROW_ATR = 0.30
    PCT_FALLBACK = 0.0015
    PCT_FALLBACK_NARROW = 0.0010
    INITIAL_SL_ATR = 2.0
    TRAIL_ATR = 2.0
    BREAKEVEN_TRIGGER_R = 1.0
    CANDLE_SL_TRIGGER_R = 1.5

    # Volatility Filter
    BB_BANDWIDTH_THRESHOLD = 0.005  # Below this = choppy market

    # Cooldown & Deduplication
    FLIP_COOLDOWN_BARS = 1
    MAX_REPEATED_EXITS_PER_BAR = 1
    DEDUPE_ONE_ENTRY_PER_BAR = True
    DEDUPE_BY_REGIME_AFTER_FAIL = True

    # Signal mode
    SIGNAL_MODE = "both"  # "both", "macd", "ema"

    # Dynamic CPR Stop Loss
    CPR_SL_BUFFER_ATR_MULTIPLIER = 0.2  # 20% of ATR as buffer
    CPR_SL_PRICE_TOLERANCE_PCT = 0.10  # Max 10% distance from CPR levels

    def __init__(self, fyers_service, symbol, lot_size, log_fn, state_path, ai_predictor, event_log=None, bot=None, **_):
        self.log        = log_fn
        self.gap = GapProtector(self.log)
        self.dynamic_sl_override = None
        self.last_trend_signal = None
        self.df = None
        self.svc        = fyers_service
        self.symbol     = symbol
        self.lot        = int(lot_size)
        self.state_path = state_path
        # --- OBX Option Engine set-up ---
        try:
            self.symbol_clean = self.symbol.replace(":", "_")
        except Exception:
            self.symbol_clean = str(self.symbol)
        try:
            self.is_option_symbol = SymbolParser.is_option(self.symbol)
        except Exception:
            s = (self.symbol or "").upper()
            self.is_option_symbol = s.endswith("CE") or s.endswith("PE")
        try:
            if self.is_option_symbol:
                self.option_engine = OptionBuyingStrategy(
                    fyers_sdk=self.svc.sdk,
                    logger=self.log,
                    flow_path=os.path.join(os.path.dirname(self.state_path), f".obx_flow_{self.symbol_clean}.json")
                )
            else:
                self.option_engine = None
        except Exception as _e_init_obx:
            self.log(f"[OBX] init error: {_e_init_obx}", False)
            self.option_engine = None

        self.event_log  = event_log
        self.bot        = bot  # Add bot reference
        self.lock       = threading.Lock()
        self.report_dir = _.get("report_dir")
        self.trades_csv = None
        self.ai_predictor = ai_predictor
        if not hasattr(self, "position"):
            self.position = {}
        self.position.setdefault("trail_active", False)
        self.position.setdefault("trail_start_profit", 5)  # Profit (in points) after which trail starts
        self.position.setdefault("trail_gap", 2)  # Lock profit (e.g. 2 points below peak)
        self.position.setdefault("max_profit", 0)
        self.position.setdefault("recent_trail_exit", False)


        # Link AI predictor to this order manager
        if self.ai_predictor:
            self.ai_predictor.order_manager = self

        if self.report_dir:
            os.makedirs(self.report_dir, exist_ok=True)
            self.trades_csv = os.path.join(
                self.report_dir,
                f"{self.symbol.replace(':','_')}_trades.csv"
            )
            self._ensure_trade_csv()

        if self.PAPER_TRADING_MODE:
            self.log("<<<<< PAPER TRADING MODE IS ACTIVE >>>>>", False)

        self._load_state()

        # Market data cache
        self.last_known_ltp = None
        self.last_known_inds = None
        self.last_known_primary_tf = None

        # AI CPR state
        self.last_ai_action = None
        self.last_ai_confidence = 0.0
        self.ai_entry_attempt_bar = None

    # ---------- CORE HELPER METHODS ----------
    @staticmethod
    def _f(v, alt=None):
        try:
            if v is None: return alt
            x = float(v)
            return x if isfinite(x) else alt
        except Exception:
            return alt

    @staticmethod
    def _cross_up(a_prev, b_prev, a, b):
        return (a_prev is not None and b_prev is not None and a is not None and b is not None
                and a_prev <= b_prev and a > b)

    @staticmethod
    def _cross_dn(a_prev, b_prev, a, b):
        return (a_prev is not None and b_prev is not None and a is not None and b is not None
                and a_prev >= b_prev and a < b)

    @staticmethod
    def _near(a, b, tol):
        return (a is not None and b is not None and abs(a - b) <= tol)

    def _atr_gap(self, a, b, atr):
        a, b, atr = self._f(a), self._f(b), self._f(atr)
        if a is None or b is None or (atr is None or atr == 0):
            return None
        return abs(a - b) / atr

    def _now_iso(self):
        try:
            return dt.datetime.now(IST).isoformat()
        except Exception:
            return dt.datetime.utcnow().isoformat()

    def _get_atr_with_fallback(self, inds_tf: dict, price: float):
        atr_val = self._f(inds_tf.get("ATR")) if isinstance(inds_tf, dict) else None
        if atr_val is None:
            atr_val = self._f(inds_tf.get("atr")) if isinstance(inds_tf, dict) else None
        if (atr_val is None or atr_val <= 0) and price is not None:
            atr_val = float(price) * self.PCT_FALLBACK
            self.log(f"[ENGINE] ATR fallback engaged: {atr_val:.6f}", True)
        return atr_val

    def _norm_tf(self, all_inds, tf: str):
        tf_str = str(tf)
        root = all_inds or {}
        if isinstance(root, list):
            pick = next((x for x in root if isinstance(x, dict) and isinstance(x.get("Dashboard"), dict)), None)
            if pick is not None:
                root = pick["Dashboard"]
            else:
                pick = next((x for x in root if isinstance(x, dict) and isinstance(x.get(tf_str), dict)), None)
                root = pick if pick is not None else {}
        if isinstance(root, dict) and isinstance(root.get("Dashboard"), dict):
            root = root["Dashboard"]
        if not isinstance(root, dict): return {}
        d = root.get(tf_str)
        if d is None: d = root.get(int(tf_str), {})
        if not isinstance(d, dict): return {}
        inds = d.get("inds")
        return inds if isinstance(inds, dict) else d

    def _load_state(self):
        self.position = robust_load_json(self.state_path, self.log, default={})
        self.position.setdefault("_last_bar_key", None)
        self.position.setdefault("_last_action_bar", None)
        self.position.setdefault("_exits_this_bar", 0)
        self.position.setdefault("initialized", False)
        self.position.setdefault("_skip_entry_until_bar", None)
        self.log(f"State loaded: Position is {self.position.get('type', 'FLAT')}", True)

    def _save_state(self):
        robust_save_json(self.position, self.state_path, self.log)
        self.log(f"State saved: Position is now {self.position.get('type', 'FLAT')}", True)

    # ---------- TRADE EXECUTION METHODS ----------
    def _process_entry(self, side, reason, ltp, atr, bar_key=None, indsP=None):

        indsP = indsP or {}
        # GAP guard (ORB/retest + gap-aware SL)
        try:
            ema5 = (indsP or {}).get("ema_5")
            ema21 = (indsP or {}).get("ema_21")
            side_word = "Buy" if str(side).upper()== "BUY" else ("Sell" if str(side).upper()== "SELL" else str(side))
            blocked, why, sl_override = self.gap.block_entry(side_word, dt.datetime.now(IST), ltp, ema5, ema21)
            if blocked:
                self.log(f"[GAP] Blocked {side_word} entry: {why}", False)
                return False
            if sl_override is not None:
                self.dynamic_sl_override = sl_override
        except Exception as _ge:
            self.log(f"[GAP] guard error: {_ge}", True)

        self.position["_last_entry_attempt_bar"] = bar_key

        if atr is None or atr <= 0:
            self.log("ENTRY BLOCKED: ATR unavailable.", False)
            return False

        resp, _ = self.svc.place_market_order(self.symbol, side, self.lot)

        order_id = None
        if isinstance(resp, dict):
            order_id = resp.get("id") or resp.get("orderId")
            if not order_id and isinstance(resp.get("data"), dict):
                order_id = resp["data"].get("id") or resp["data"].get("orderId")

        # success or paper-override on margin shortfall patterns
        is_live_success = isinstance(resp, dict) and resp.get("s") == "ok" and order_id
        is_paper_override = (
                self.PAPER_TRADING_MODE and isinstance(resp, dict) and
                (resp.get("code") == -99 or "margin" in str(resp.get("message", "")).lower())
        )

        if is_live_success or is_paper_override:
            if is_paper_override:
                self.log("PAPER TRADE: Margin error detected. Simulating successful entry.")
                order_id = order_id or f"PAPER-{int(time.time())}"

            r_mult = self.INITIAL_SL_ATR * atr
            # Allow optional gap-aware override
            if self.dynamic_sl_override is not None:
                initial_sl = float(self.dynamic_sl_override)
                self.dynamic_sl_override = None
            else:
                initial_sl = ltp - r_mult if side == "BUY" else ltp + r_mult
            now_iso = self._now_iso()

            # Round initial SL to nearest exchange tick size to avoid rejections
            try:
                ts = get_tick_size(self.svc.sdk, self.symbol)
            except Exception:
                ts = None
            if ts and ts > 0:
                initial_sl = round(round(initial_sl / ts) * ts, 2)

            # 🔥 Extract AI confidence for logging
            ai_confidence = self.position.get("ai_entry_confidence", 0.0)

            # Save position state
            self.position = {
                "type": side,
                "order_id": order_id,
                "entry_price": ltp,
                "stop_loss": initial_sl,
                "r_mult": r_mult,
                "breakeven_set": False,
                "ts": now_iso,
                "max_profit": 0.0,
                "trail_active": False,
                "ai_entry_confidence": ai_confidence,  # 🔥 Store AI confidence
                "_last_bar_key": self.position.get("_last_bar_key"),
                "_last_action_bar": bar_key,
                "_exits_this_bar": 0
            }

            # 🔥 Enhanced logging with AI confidence
            self.log(
                f"{'PAPER ' if is_paper_override else ''}ENTRY SUCCESS: {side} {self.symbol}. "
                f"Reason: {reason}. Entry: {ltp:.2f} | Initial SL: {initial_sl:.2f} | 1R={r_mult:.2f}"
                f"{f' | AI Confidence: {ai_confidence:.3f}' if ai_confidence > 0 else ''}",
                False
            )

            self._append_trade_csv({
                "trade_id": order_id, "symbol": self.symbol, "side": side, "event": "ENTRY",
                "entry_time": now_iso, "entry_ltp": ltp, "reason": reason, "order_id": order_id,
                "bar_key": bar_key,
                "adx": self._f(indsP.get("adx")),
                "macd_color": indsP.get("macd_color"),
                "ema5": self._f(indsP.get("ema_5")),
                "ema9": self._f(indsP.get("ema_9")),
                "ema21": self._f(indsP.get("ema_21")),
                "ai_confidence": ai_confidence  # 🔥 Log to CSV
            })
            self.log(f"DEBUG: ENTRY event logged to CSV for {side} {self.symbol}", True)
            self._save_state()
            return True

        # Genuine failure
        msg = (resp or {}).get("message", "Unknown Error")
        self.log(f"ENTRY FAILED: {msg}", False)
        self._append_trade_csv({
            "trade_id": f"{self.symbol}-{int(time.time())}",
            "symbol": self.symbol,
            "side": side,
            "event": "ENTRY_FAIL",
            "entry_time": self._now_iso(),
            "entry_ltp": ltp,
            "reason": msg,
            "order_id": (resp or {}).get("id"),
            "bar_key": bar_key,
            "adx": self._f(indsP.get("adx")),
            "macd_color": indsP.get("macd_color"),
            "ema5": self._f(indsP.get("ema_5")),
            "ema9": self._f(indsP.get("ema_9")),
            "ema21": self._f(indsP.get("ema_21"))
        })
        self._save_state()
        return False


    def _process_exit(self, reason, ltp):
        """
        Simulates exits in paper mode; otherwise sends exit to broker.
        """
        if not self.position or self.position.get("type") == "FLAT":
            return True

        side      = self.position.get("type")
        entry_ltp = self._f(self.position.get("entry_price"))
        entry_ts  = self.position.get("ts")
        order_id  = self.position.get("order_id")

        resp = None
        if self.PAPER_TRADING_MODE:
            self.log(f"PAPER TRADE: Simulating exit for {self.symbol}. Reason: {reason}")
            resp = {"s": "ok", "code": 200, "message": "Paper Trade Exit"}
        else:
            resp, _ = self.svc.exit_all_positions_for_symbol(self.symbol)

        if isinstance(resp, dict) and (resp.get("s") == "ok" or resp.get("code") in [-66, 204]):
            now_iso = self._now_iso()
            log_msg_prefix = "PAPER EXIT SUCCESS" if self.PAPER_TRADING_MODE else "EXIT SUCCESS"
            self.log(f"{log_msg_prefix}. Reason: {reason}")

            hold_sec = 0
            try:
                if entry_ts:
                    t0 = pd.to_datetime(entry_ts)
                    t1 = pd.to_datetime(now_iso)
                    hold_sec = max(0, int((t1 - t0).total_seconds()))
            except Exception:
                pass

            ltp_diff = None
            if entry_ltp is not None and ltp is not None:
                ltp_diff = (ltp - entry_ltp) if side == "BUY" else (entry_ltp - ltp)

            try:
                self._append_trade_csv({
                    "trade_id": order_id, "symbol": self.symbol, "side": side, "event": "EXIT",
                    "entry_time": entry_ts, "exit_time": now_iso,
                    "hold_seconds": hold_sec,
                    "entry_ltp": entry_ltp, "exit_ltp": ltp, "ltp_diff": ltp_diff,
                    "reason": reason, "order_id": order_id,
                    "bar_key": self.position.get("_last_bar_key", "")
                })
                self.log(f"DEBUG: EXIT event logged to CSV for {side} {self.symbol}", True)
            except Exception as e:
                self.log(f"[ERROR] Failed to log EXIT to CSV in _process_exit: {e}", False)

            self.position["last_type"] = side  # Record the last exited position type

            self.position["type"] = "FLAT"
            self.position["exit_price"] = ltp
            self.position["exit_ts"] = now_iso
            # Reset state after exit
            self.position = {
                "_last_bar_key": self.position.get("_last_bar_key"),
                "_last_action_bar": self.position.get("_last_action_bar"),
                "_last_entry_attempt_bar": self.position.get("_last_entry_attempt_bar"),
                "_exits_this_bar": self.position.get("_exits_this_bar", 0),
                "_cooldown_bars": self.FLIP_COOLDOWN_BARS,
                "_last_exit_side": side,
                "type": "FLAT"
            }
            self.position["_skip_entry_until_bar"] = self.position.get("_last_bar_key") #Kiran added
            self._save_state()
            return True

        self.log(f"EXIT FAILED: {resp.get('message', 'Unknown Error') if isinstance(resp, dict) else 'No response'}")
        return False

    def _is_macd_flipped(self, inds):
        if not inds or "MACD" not in inds or "MACD_SIGNAL" not in inds:
            return False
        return (
                (self.position["type"] == "BUY" and inds["MACD"] < inds["MACD_SIGNAL"]) or
                (self.position["type"] == "SELL" and inds["MACD"] > inds["MACD_SIGNAL"])
        )

    def _is_exit_signal(self, inds):
        if not inds:
            return False
        st = inds.get("SUPERTREND")
        ltp = inds.get("LTP")
        if st is None or ltp is None:
            return False

        if self.position["type"] == "BUY" and ltp < st:
            return True
        if self.position["type"] == "SELL" and ltp > st:
            return True
        return False

    def _check_trailing_profit(self, ltp, inds=None):
        pos = self.position
        if not pos or pos.get("type") == "FLAT":
            return

        entry = pos.get("entry_price")
        if not entry:
            return

        side = pos["type"]
        profit = (ltp - entry) if side == "BUY" else (entry - ltp)
        max_profit = pos.get("_max_profit", 0)
        if profit > max_profit:
            pos["_max_profit"] = profit
            max_profit = profit
        drawdown = max_profit - profit if max_profit > 0 else 0

        atr_val = self._get_atr_with_fallback(inds, ltp) if inds else None
        if atr_val is None or atr_val <= 0:
            atr_val = 1.5

        # === Trend strength detection (adaptive & defensive) ===
        trend_strong = False
        trend_very_strong = False

        self.log(
            f"[DEBUG] _check_trailing_profit called — inds={type(inds)} keys={list(inds.keys())[:10] if inds and isinstance(inds, dict) else None}",
            True
        )

        if inds and isinstance(inds, dict):
            e5 = self._f(inds.get("ema_5"))
            e9 = self._f(inds.get("ema_9"))
            e21 = self._f(inds.get("ema_21"))
            adx = self._f(inds.get("adx"))
            macd_color = str(inds.get("macd_color", "")).strip().lower()
            bb_bandwidth = self._f(inds.get("bb_bandwidth"))
            supertrend = self._f(inds.get("supertrend"))

            # ✅ More defensive price_above_st check
            price_above_st = False
            if supertrend is not None and ltp is not None:
                try:
                    price_above_st = float(ltp) > float(supertrend)
                except (ValueError, TypeError):
                    pass

            self.log(
                f"[DEBUG] Trend check — e5={e5}, e9={e9}, e21={e21}, "
                f"adx={adx}, macd_color='{macd_color}', bb_bw={bb_bandwidth}, "
                f"supertrend={supertrend}, price_above_st={price_above_st}",
                True
            )

            def macd_is_bullish(color):
                if not color:
                    return False
                return any(x in color for x in ["green", "up", "bull"])

            def macd_is_bearish(color):
                if not color:
                    return False
                return any(x in color for x in ["red", "down", "bear"])

            # ✅ More lenient trend detection (was failing too often)
            if side == "BUY":
                # Basic trend check
                basic_trend_ok = (
                        e5 is not None and e21 is not None and e5 > e21 * 0.998  # More lenient (was 0.999)
                )
                # Momentum confirmation (need at least 1)
                momentum_ok = (
                        (adx is not None and adx > 18) or
                        price_above_st or
                        macd_is_bullish(macd_color)
                )

                trend_strong = basic_trend_ok and momentum_ok

                # Very strong requires all aligned
                trend_very_strong = (
                        trend_strong and
                        e9 is not None and e5 > e9 > e21 and
                        adx is not None and adx > 30 and
                        macd_color and "dark green" in macd_color  # Strongest MACD
                )

            elif side == "SELL":
                basic_trend_ok = (
                        e5 is not None and e21 is not None and e5 < e21 * 1.002  # More lenient
                )
                momentum_ok = (
                        (adx is not None and adx > 18) or
                        (supertrend is not None and ltp < supertrend) or
                        macd_is_bearish(macd_color)
                )

                trend_strong = basic_trend_ok and momentum_ok

                trend_very_strong = (
                        trend_strong and
                        e9 is not None and e5 < e9 < e21 and
                        adx is not None and adx > 30 and
                        macd_color and "dark red" in macd_color
                )

            self.log(
                f"[TREND RESULT] Strong={trend_strong}, VeryStrong={trend_very_strong}, "
                f"BasicTrend={basic_trend_ok if 'basic_trend_ok' in locals() else 'N/A'}, "
                f"Momentum={momentum_ok if 'momentum_ok' in locals() else 'N/A'}",
                True
            )
        else:
            self.log(f"[WARN] No valid indicators - defaulting to WEAK trend", True)

        # === Progressive targets (same logic as before) ===
        if atr_val < 1.0:
            base_multiplier = 1.0
        elif atr_val < 2.0:
            base_multiplier = 1.5
        else:
            base_multiplier = 2.0

        profit_tier = pos.get("_profit_tier", 0)
        if trend_very_strong:
            tier_multipliers = [base_multiplier * 2.0, base_multiplier * 3.0, base_multiplier * 4.0]
            tier_label = "VERY_STRONG"
            trailing_pct = 0.20
        elif trend_strong:
            tier_multipliers = [base_multiplier * 1.5, base_multiplier * 2.5]
            tier_label = "STRONG"
            trailing_pct = 0.25
        else:
            tier_multipliers = [base_multiplier]
            tier_label = "WEAK"
            trailing_pct = 0.30

        current_tier = min(profit_tier, len(tier_multipliers) - 1)
        profit_target = tier_multipliers[current_tier] * atr_val

        # Move to next profit tier dynamically
        if profit >= profit_target and current_tier < len(tier_multipliers) - 1:
            pos["_profit_tier"] = current_tier + 1
            next_target = tier_multipliers[current_tier + 1] * atr_val
            self.log(
                f"🎯 TIER UP! Profit ₹{profit:.2f} hit {profit_target:.2f}. "
                f"Next target ₹{next_target:.2f} (Trend: {tier_label})", False
            )
            profit_target = next_target
            current_tier += 1

        # ==========================================
        # ✅ Exit logic
        # ==========================================
        if profit >= profit_target:
            if trend_strong or trend_very_strong:
                if drawdown >= trailing_pct * max_profit:
                    self.log(
                        f"⚠️ {tier_label} trend trailing stop: Profit dropped {drawdown:.2f} "
                        f"({trailing_pct * 100:.0f}%) from peak ₹{max_profit:.2f} — exiting.", False
                    )
                    try:
                        self._process_exit(f"Trailing Stop ({tier_label} Trend, Tier {current_tier + 1})", ltp)
                    except Exception as e:
                        self.log(f"[ERROR] Trailing exit failed: {e}")
                    return
                else:
                    self.log(
                        f"💎 HOLDING {tier_label} trend — Profit: ₹{profit:.2f}, "
                        f"Target: ₹{profit_target:.2f}, Peak: ₹{max_profit:.2f}, "
                        f"Tier: {current_tier + 1}/{len(tier_multipliers)}", True
                    )
            else:
                self.log(f"💰 Profit target ₹{profit:.2f} reached in weak trend — exiting!", False)
                try:
                    self._process_exit("Profit Target Reached (Weak Trend)", ltp)
                except Exception as e:
                    self.log(f"[ERROR] Profit target exit failed: {e}")
                return

        # Early profit protection
        if max_profit >= profit_target * 0.4 and drawdown >= 0.35 * max_profit:
            self.log(
                f"⚠️ Early profit protection: dropped {drawdown:.2f} ({drawdown / max_profit * 100:.0f}%) "
                f"from ₹{max_profit:.2f} — exiting.", False
            )
            try:
                self._process_exit("Early Profit Protection", ltp)
            except Exception as e:
                self.log(f"[ERROR] Drawdown exit failed: {e}")
            return

        # ✅ Improved trend reversal detection
        if inds and isinstance(inds, dict) and max_profit > 0:
            e5 = self._f(inds.get("ema_5"))
            e21 = self._f(inds.get("ema_21"))
            macd_color = str(inds.get("macd_color", "")).strip().lower()
            adx = self._f(inds.get("adx"))

            # ✅ More robust reversal check
            trend_reversed = False
            reversal_confidence = 0  # Track how strong the reversal is

            if side == "BUY":
                # Check multiple reversal signals
                ema_reversed = (e5 is not None and e21 is not None and e5 < e21 * 0.998)
                macd_reversed = macd_color and any(x in macd_color for x in ["dark red", "light red", "red"])
                weak_momentum = (adx is not None and adx < 20)

                # Count reversal signals
                if ema_reversed:
                    reversal_confidence += 2  # EMA is strongest signal
                if macd_reversed:
                    reversal_confidence += 1
                if weak_momentum:
                    reversal_confidence += 1

                # Need at least 2 reversal signals (out of 4 points possible)
                trend_reversed = reversal_confidence >= 2

                self.log(
                    f"[REVERSAL CHECK BUY] EMA_rev={ema_reversed}, MACD_rev={macd_reversed}, "
                    f"Weak_ADX={weak_momentum}, Confidence={reversal_confidence}/4",
                    True
                )

            elif side == "SELL":
                ema_reversed = (e5 is not None and e21 is not None and e5 > e21 * 1.002)
                macd_reversed = macd_color and any(x in macd_color for x in ["dark green", "light green", "green"])
                weak_momentum = (adx is not None and adx < 20)

                if ema_reversed:
                    reversal_confidence += 2
                if macd_reversed:
                    reversal_confidence += 1
                if weak_momentum:
                    reversal_confidence += 1

                trend_reversed = reversal_confidence >= 2

                self.log(
                    f"[REVERSAL CHECK SELL] EMA_rev={ema_reversed}, MACD_rev={macd_reversed}, "
                    f"Weak_ADX={weak_momentum}, Confidence={reversal_confidence}/4",
                    True
                )

            if trend_reversed:
                self.log(
                    f"🔴 Trend REVERSED (confidence: {reversal_confidence}/4) at profit ₹{profit:.2f} — exiting immediately!",
                    False
                )
                try:
                    self._process_exit(f"Trend Reversal Exit (conf: {reversal_confidence}/4)", ltp)
                except Exception as e:
                    self.log(f"[ERROR] Trend reversal exit failed: {e}")
                return
            else:
                self.log(
                    f"[REVERSAL] Not reversed yet (confidence: {reversal_confidence}/4 < 2 needed)",
                    True
                )

        # Trailing status
        self.log(
            f"[TRAILING] {tier_label} trend — LTP={ltp}, Profit=₹{profit:.2f}, "
            f"Peak=₹{max_profit:.2f}, Target=₹{profit_target:.2f}, "
            f"Tier={current_tier + 1}/{len(tier_multipliers)}, Trail={trailing_pct * 100:.0f}%",
            True
        )

        self._save_state()

    # def _check_trailing_profit(self, ltp, inds=None):
    #     """
    #     Monitors open position profit and exits early under certain conditions:
    #     - For GOLDPETAL: locks profit aggressively once above threshold.
    #     - For others: exits if profit drawdown > 35% of peak or if trend/MACD confirm exit.
    #     """
    #     pos = self.position
    #     if not pos or pos.get("type") == "FLAT":
    #         return

    #     entry = pos.get("entry_price")
    #     if not entry:
    #         return

    #     side = pos["type"]
    #     symbol = self.symbol.upper()
    #     profit = (ltp - entry) if side == "BUY" else (entry - ltp)
    #     max_profit = pos.get("_max_profit", 0)

    #     # Update max profit tracker
    #     if profit > max_profit:
    #         pos["_max_profit"] = profit
    #         max_profit = profit

    #     # Calculate drawdown from max profit
    #     drawdown = max_profit - profit if max_profit > 0 else 0

    #     # 🟡 --- GOLDPETAL aggressive profit booking ---
    #     if "NSE:CIPLA-EQ" in symbol:
    #         if profit >= 10:  # or use dynamic 60–65% threshold if needed
    #             self.log(f"💰 GOLDPETAL profit ₹{profit:.2f} reached — exiting early!")
    #             try:
    #                 self._process_exit("GP-Profit Target Reached", ltp)  # exits and updates state
    #             except Exception as e:
    #                 self.log(f"[ERROR] GOLDPETAL exit failed: {e}")
    #             return

    #         # If profit drops more than 30% from its peak, exit to protect
    #         if max_profit > 0 and drawdown >= 0.2 * max_profit:
    #             self.log(f"⚠️ GOLDPETAL profit dropped {drawdown:.2f} from ₹{max_profit:.2f} — exiting.")
    #             try:
    #                 self._process_exit("GP-Profit going down", ltp)
    #             except Exception as e:
    #                 self.log(f"[ERROR] GOLDPETAL drawdown exit failed: {e}")
    #             return

    #     # 🔵 --- Other symbols: profit protection & MACD/trend confirmation ---
    #     else:
    #         # Early protection if drawdown > 35%
    #         if max_profit > 0 and drawdown >= 0.25 * max_profit:
    #             if not self._is_macd_flipped(inds):
    #                 self.log(
    #                     f"⚠️ Profit dropped {drawdown:.2f} from peak ₹{max_profit:.2f}, MACD still not flipped — exiting.")
    #                 try:
    #                     self._process_exit("Profit Protected and got exit", ltp)
    #                 except Exception as e:
    #                     self.log(f"[ERROR] Drawdown exit failed: {e}")
    #                 return

    #         # Normal exit condition when MACD + trend agree
    #         if inds and self._is_exit_signal(inds):
    #             self.log(f"📉 Exit confirmed by MACD & Trend — closing position.")
    #             try:
    #                 self._process_exit("MACD closed the position", ltp)
    #             except Exception as e:
    #                 self.log(f"[ERROR] Exit failed: {e}")
    #             return

    #     # Log trailing status
    #     self.log(
    #         f"[TRAILING] Active — LTP={ltp}, MaxProfit={pos.get('_max_profit', 0):.2f}, TrailSL={pos.get('trail_sl')}")
    #     self._save_state()

    def _update_trailing_and_breakeven(self, ltp, atr):
        if not self.position or self.position.get("type") in (None, "FLAT"):
            return
        side   = self.position.get("type")
        cur_sl = self._f(self.position.get("stop_loss"), 0.0)
        entry  = self._f(self.position.get("entry_price"))
        r      = self._f(self.position.get("r_mult"))
        if entry is None or r is None or atr is None or atr <= 0:
            return

        if not self.position.get("breakeven_set", False):
            if (side == "BUY" and ltp >= entry + self.BREAKEVEN_TRIGGER_R * r) or \
               (side == "SELL" and ltp <= entry - self.BREAKEVEN_TRIGGER_R * r):
                self.position["stop_loss"] = entry
                self.position["breakeven_set"] = True
                self.log(f"Breakeven set at {entry:.2f} after +{self.BREAKEVEN_TRIGGER_R:.1f}R", True)
                return

        new_tsl = (ltp - self.TRAIL_ATR * atr) if side == "BUY" else (ltp + self.TRAIL_ATR * atr)
        if (side == "BUY" and new_tsl > cur_sl) or (side == "SELL" and new_tsl < cur_sl):
            self.position["stop_loss"] = new_tsl
            self.log(f"Trailing SL → {new_tsl:.2f}", True)

    # ---------- AI CPR ORDER METHODS ----------
    def ai_buy(self, symbol, qty):
        """Execute AI-driven BUY order"""
        self.log(f"[AI-CPR] Executing BUY order for {symbol}, qty: {qty}", False)
        return self.place_order(symbol, qty, side="BUY", tag="AI-CPR")

    def ai_sell(self, symbol, qty):
        """Execute AI-driven SELL order"""
        self.log(f"[AI-CPR] Executing SELL order for {symbol}, qty: {qty}", False)
        return self.place_order(symbol, qty, side="SELL", tag="AI-CPR")

    def ai_exit_all(self, symbol):
        """Execute AI-driven exit for all positions"""
        self.log(f"[AI-CPR] Exiting all positions for {symbol}", False)
        current_ltp_for_exit = self.last_known_ltp if self.last_known_ltp is not None else 0.0
        if current_ltp_for_exit == 0.0:
            self.log("[AI-CPR] Warning: Current LTP for AI exit is 0.0", False)
        return self._process_exit(reason="AI-CPR requested exit", ltp=current_ltp_for_exit)

    def execute_ai_cpr_strategy(self, ltp, all_inds, primary_tf="5"):
        """
        Execute AI CPR-based trading strategy independently
        """
        if not self.AI_CPR_ENABLED:
            return

        self.last_known_ltp = ltp
        self.last_known_inds = all_inds
        self.last_known_primary_tf = primary_tf

        def _tf(tf):
            return self._norm_tf(all_inds, str(tf))

        inds = _tf(primary_tf)
        if not isinstance(inds, dict) or not inds.get("timestamp"):
            self.log(f"[AI-CPR] No indicators for TF={primary_tf}.", True)
            return

        # Bar key for deduplication
        bar_key = f"{primary_tf}:{inds['timestamp']}"

        # ─── GAP state maintenance ─────────────────────────────────────
        try:
            cpr_analysis = inds.get("cpr_analysis", {}) if isinstance(inds, dict) else {}
            pivot_data = cpr_analysis.get("cpr_levels", {}) if isinstance(cpr_analysis, dict) else {}
            ohlc_gap = None
            try:
                if hasattr(self, 'bot') and callable(getattr(self.bot, 'fetch_ohlc', None)):
                    ohlc_gap = self.bot.fetch_ohlc(self.symbol, str(primary_tf), 60)
            except Exception as _e_fetch_gap:
                self.log(f"[GAP] OHLC fetch err: {_e_fetch_gap}", True)
            try:
                import pandas as _pd_gap
                ts_now = _pd_gap.to_datetime(inds.get('timestamp')).to_pydatetime()
            except Exception:
                ts_now = dt.datetime.now(IST)
            self.gap.reset_if_new_session(ts_now, ohlc_gap, pivot_data)
            self.gap.update_opening_range(ts_now, ohlc_gap)
        except Exception as _e_gmk:
            self.log(f"[GAP] setup error: {_e_gmk}", True)

        # Fast exit if gap fails
        if self.position.get('type') in ('BUY','SELL'):
            try:
                side_lbl = 'LONG' if self.position.get('type') == 'BUY' else 'SHORT'
                try:
                    close_px = float(inds.get('close'))
                except Exception:
                    close_px = ltp
                do_exit, msg = self.gap.gap_fail_exit(side_lbl, close_px)
                if do_exit:
                    self._process_exit(f"GapFail: {msg}", ltp)
                    return
            except Exception as _e_gfail:
                self.log(f"[GAP] fail-exit check err: {_e_gfail}", True)

        # Kiran Added this 3 lines
        if self.DEDUPE_ONE_ENTRY_PER_BAR and self.position.get("_last_entry_attempt_bar") == bar_key:
            self.log("AI entry blocked - already attempted this bar", True)
            return
        # --- ⛔ Re-entry Cooldown Check ---
        if self.position.get("_skip_entry_until_bar") == bar_key:
            self.log("[RE-ENTRY] Skipping new trade this bar (1-bar cooldown active)", True)
            return

        # Prevent multiple AI entries in same bar
        if self.ai_entry_attempt_bar == bar_key:
            return

        # Get CPR analysis
        cpr_analysis = inds.get("cpr_analysis", {})
        ai_label = cpr_analysis.get("ai_cpr_label")
        ai_confidence = cpr_analysis.get("ai_confidence", 0.0)
        ai_filter_pass = cpr_analysis.get("ai_filter_pass", True)

        # Update AI state
        self.last_ai_action = ai_label
        self.last_ai_confidence = ai_confidence

        # Execute AI trading logic
        if ai_label and ai_confidence and ai_confidence >= self.AI_MIN_CONF:
            action = ai_label.upper()

            if action == "BUY" and ai_filter_pass:
                self.log(f"[AI-CPR] STRONG BUY signal (confidence: {ai_confidence:.3f})", False)
                if self.position.get("type") != "BUY":
                    self.ai_exit_all(self.symbol)
                    if self.ai_buy(self.symbol, self.lot):
                        self.ai_entry_attempt_bar = bar_key
                        self._log_ai_trade("BUY", ai_confidence, cpr_analysis.get("reason", "AI Signal"))

            elif action == "SELL" and ai_filter_pass:
                self.log(f"[AI-CPR] STRONG SELL signal (confidence: {ai_confidence:.3f})", False)
                if self.position.get("type") != "SELL":
                    self.ai_exit_all(self.symbol)
                    if self.ai_sell(self.symbol, self.lot):
                        self.ai_entry_attempt_bar = bar_key
                        self._log_ai_trade("SELL", ai_confidence, cpr_analysis.get("reason", "AI Signal"))

            elif action in ["HOLD", "NEUTRAL"]:
                self.log(f"[AI-CPR] HOLD signal (confidence: {ai_confidence:.3f})", True)
                # Optionally exit positions on strong HOLD signal
                if self.position.get("type") != "FLAT" and ai_confidence > 0.7:
                    self.log(f"[AI-CPR] Exiting position due to strong HOLD signal", False)
                    self.ai_exit_all(self.symbol)

    def _log_ai_trade(self, side, confidence, reason):
        """Log AI-specific trade details"""
        try:
            self._append_trade_csv({
                "trade_id": f"AI-{int(time.time())}",
                "symbol": self.symbol,
                "side": side,
                "event": f"AI_{side}",
                "entry_time": self._now_iso(),
                "entry_ltp": self.last_known_ltp,
                "reason": f"AI-CPR: {reason} (conf: {confidence:.3f})",
                "order_id": "AI-GENERATED",
                "bar_key": self.position.get("_last_bar_key", ""),
                "ai_confidence": confidence,
                "ai_action": side
            })
        except Exception as e:
            self.log(f"[AI-CPR] Failed to log AI trade: {e}", False)

    # ---------- UNIFIED ORDER PLACEMENT ----------
    def place_order(self, symbol, qty, side, tag):
        """
        Unified order placement method for both manual and AI orders
        """
        current_ltp = self.last_known_ltp
        current_inds = self.last_known_inds
        primary_tf = self.last_known_primary_tf

        if current_ltp is None or current_inds is None or primary_tf is None:
            self.log(f"[{tag}] Cannot place {side} order for {symbol} - missing market data", False)
            return False

        # Get ATR for SL calculation
        atr_val = self._get_atr_with_fallback(self._norm_tf(current_inds, primary_tf), current_ltp)
        if atr_val is None or atr_val <= 0:
            self.log(f"[{tag}] Cannot place {side} order for {symbol} - ATR unavailable", False)
            return False

        reason_str = f"{tag} {side} Signal"
        bar_key = self.position.get("_last_bar_key")
        self.log(f"[AI-Order] Computed ATR: {atr_val}", True)

        # Route to entry processing
        if side == "BUY":
            return self._process_entry("BUY", reason_str, current_ltp, atr_val,
                                     bar_key=bar_key, indsP=self._norm_tf(current_inds, primary_tf))
        elif side == "SELL":
            return self._process_entry("SELL", reason_str, current_ltp, atr_val,
                                     bar_key=bar_key, indsP=self._norm_tf(current_inds, primary_tf))
        else:
            self.log(f"[{tag}] Unknown order side: {side}", False)
            return False

    def _update_dynamic_cpr_stop_loss(self, ltp, atr, pivot_data):

        # ==========================================
        # CRITICAL VALIDATION
        # ==========================================

        # Validate position exists
        if not self.position or self.position.get("type") == "FLAT":
            return

        # Validate inputs
        if ltp is None or ltp <= 0:
            self.log(f"[CPR-SL] Invalid LTP: {ltp} - skipping dynamic SL", True)
            return

        if atr is None or atr <= 0:
            self.log(f"[CPR-SL] Invalid ATR: {atr} - skipping dynamic SL", True)
            return

        if not pivot_data or not isinstance(pivot_data, dict):
            self.log("[CPR-SL] Invalid pivot data - skipping dynamic SL", True)
            return

        # Extract position details
        side = self.position.get("type")
        current_sl = self._f(self.position.get("stop_loss"))
        entry = self._f(self.position.get("entry_price"))

        if entry is None or current_sl is None:
            self.log("[CPR-SL] Missing entry or stop loss - skipping dynamic SL", True)
            return

        # Validate side
        if side not in ("BUY", "SELL"):
            self.log(f"[CPR-SL] Invalid position side: {side} - skipping dynamic SL", True)
            return

        # ==========================================
        # EXTRACT & VALIDATE CPR LEVELS
        # ==========================================

        tc = self._f(pivot_data.get("TC"))
        bc = self._f(pivot_data.get("BC"))
        r1 = self._f(pivot_data.get("R1"))
        r2 = self._f(pivot_data.get("R2"))
        r3 = self._f(pivot_data.get("R3"))
        s1 = self._f(pivot_data.get("S1"))
        s2 = self._f(pivot_data.get("S2"))
        s3 = self._f(pivot_data.get("S3"))

        # Validate core CPR levels exist
        if tc is None or bc is None:
            self.log("[CPR-SL] TC or BC missing - skipping dynamic SL", True)
            return

        # Sanity check: TC should be above BC
        if tc < bc:
            self.log(f"[CPR-SL] WARNING: TC ({tc}) < BC ({bc}) - inverted CPR! Skipping dynamic SL", False)
            return

        # Check if CPR levels are reasonable relative to price (within 10%)
        if abs(ltp - tc) > ltp * 0.10:
            self.log(
                f"[CPR-SL] WARNING: Price {ltp:.2f} too far from TC {tc:.2f} ({abs(ltp - tc) / ltp * 100:.1f}% away) - CPR might be stale",
                True
            )
            # Still continue, but log warning

        # ==========================================
        # CALCULATE BUFFER (prevents premature stops)
        # ==========================================

        buffer = 0.2 * atr  # 20% of ATR as safety buffer

        # ==========================================
        # LONG POSITION LOGIC
        # ==========================================

        if side == "BUY":
            self.log(
                f"[CPR-SL] LONG - Price: {ltp:.2f}, Current SL: {current_sl:.2f}, Entry: {entry:.2f}",
                True
            )

            # Priority 1: If crossed R3, move SL to R2
            if r3 and r2 and ltp > r3:
                new_sl = r2 - buffer
                if new_sl > current_sl:
                    self.position["stop_loss"] = round(new_sl, 2)
                    profit_locked = new_sl - entry
                    self.log(
                        f"[CPR-SL] ✅ LONG SL raised to R2: {new_sl:.2f} (price above R3 at {r3:.2f}) | "
                        f"Profit locked: {profit_locked:.2f} points",
                        False
                    )
                    self._save_state()
                    return
                else:
                    self.log(
                        f"[CPR-SL] LONG - R3 crossed but new SL {new_sl:.2f} not higher than current {current_sl:.2f} - skipping",
                        True
                    )

            # Priority 2: If crossed R2, move SL to R1
            elif r2 and r1 and ltp > r2:
                new_sl = r1 - buffer
                if new_sl > current_sl:
                    self.position["stop_loss"] = round(new_sl, 2)
                    profit_locked = new_sl - entry
                    self.log(
                        f"[CPR-SL] ✅ LONG SL raised to R1: {new_sl:.2f} (price above R2 at {r2:.2f}) | "
                        f"Profit locked: {profit_locked:.2f} points",
                        False
                    )
                    self._save_state()
                    return
                else:
                    self.log(
                        f"[CPR-SL] LONG - R2 crossed but new SL {new_sl:.2f} not higher than current {current_sl:.2f} - skipping",
                        True
                    )

            # Priority 3: If crossed R1, move SL to TC
            elif r1 and tc and ltp > r1:
                new_sl = tc - buffer
                if new_sl > current_sl:
                    self.position["stop_loss"] = round(new_sl, 2)
                    profit_locked = new_sl - entry
                    profit_status = "✅ RISK-FREE" if new_sl >= entry else "⚠️ Still at risk"
                    self.log(
                        f"[CPR-SL] ✅ LONG SL raised to TC: {new_sl:.2f} (price above R1 at {r1:.2f}) | "
                        f"Profit locked: {profit_locked:.2f} points | {profit_status}",
                        False
                    )
                    self._save_state()
                    return
                else:
                    self.log(
                        f"[CPR-SL] LONG - R1 crossed but new SL {new_sl:.2f} not higher than current {current_sl:.2f} - skipping",
                        True
                    )

            # Priority 4: If above TC (and entered below TC), move SL to BC (conservative)
            elif tc and bc and ltp > tc and entry < tc:
                new_sl = bc - buffer
                if new_sl > current_sl:
                    self.position["stop_loss"] = round(new_sl, 2)
                    risk_reduction = entry - new_sl
                    original_risk = entry - current_sl
                    reduction_pct = (original_risk - risk_reduction) / original_risk * 100 if original_risk > 0 else 0
                    self.log(
                        f"[CPR-SL] ✅ LONG SL raised to BC: {new_sl:.2f} (price above TC at {tc:.2f}) | "
                        f"Risk reduced by {reduction_pct:.1f}% (from {original_risk:.2f} to {risk_reduction:.2f} points)",
                        False
                    )
                    self._save_state()
                    return
                else:
                    self.log(
                        f"[CPR-SL] LONG - Above TC but new SL {new_sl:.2f} not higher than current {current_sl:.2f} - skipping",
                        True
                    )

            else:
                self.log(
                    f"[CPR-SL] LONG - No CPR level adjustments triggered | "
                    f"Price: {ltp:.2f}, TC: {tc}, R1: {r1}, R2: {r2}, R3: {r3}",
                    True
                )

        # ==========================================
        # SHORT POSITION LOGIC (Symmetrical)
        # ==========================================

        elif side == "SELL":
            self.log(
                f"[CPR-SL] SHORT - Price: {ltp:.2f}, Current SL: {current_sl:.2f}, Entry: {entry:.2f}",
                True
            )

            # Priority 1: If crossed S3, move SL to S2
            if s3 and s2 and ltp < s3:
                new_sl = s2 + buffer
                if new_sl < current_sl:
                    self.position["stop_loss"] = round(new_sl, 2)
                    profit_locked = entry - new_sl
                    self.log(
                        f"[CPR-SL] ✅ SHORT SL lowered to S2: {new_sl:.2f} (price below S3 at {s3:.2f}) | "
                        f"Profit locked: {profit_locked:.2f} points",
                        False
                    )
                    self._save_state()
                    return
                else:
                    self.log(
                        f"[CPR-SL] SHORT - S3 crossed but new SL {new_sl:.2f} not lower than current {current_sl:.2f} - skipping",
                        True
                    )

            # Priority 2: If crossed S2, move SL to S1
            elif s2 and s1 and ltp < s2:
                new_sl = s1 + buffer
                if new_sl < current_sl:
                    self.position["stop_loss"] = round(new_sl, 2)
                    profit_locked = entry - new_sl
                    self.log(
                        f"[CPR-SL] ✅ SHORT SL lowered to S1: {new_sl:.2f} (price below S2 at {s2:.2f}) | "
                        f"Profit locked: {profit_locked:.2f} points",
                        False
                    )
                    self._save_state()
                    return
                else:
                    self.log(
                        f"[CPR-SL] SHORT - S2 crossed but new SL {new_sl:.2f} not lower than current {current_sl:.2f} - skipping",
                        True
                    )

            # Priority 3: If crossed S1, move SL to BC
            elif s1 and bc and ltp < s1:
                new_sl = bc + buffer
                if new_sl < current_sl:
                    self.position["stop_loss"] = round(new_sl, 2)
                    profit_locked = entry - new_sl
                    profit_status = "✅ RISK-FREE" if new_sl <= entry else "⚠️ Still at risk"
                    self.log(
                        f"[CPR-SL] ✅ SHORT SL lowered to BC: {new_sl:.2f} (price below S1 at {s1:.2f}) | "
                        f"Profit locked: {profit_locked:.2f} points | {profit_status}",
                        False
                    )
                    self._save_state()
                    return
                else:
                    self.log(
                        f"[CPR-SL] SHORT - S1 crossed but new SL {new_sl:.2f} not lower than current {current_sl:.2f} - skipping",
                        True
                    )

            # Priority 4: If below BC (and entered above BC), move SL to TC (conservative)
            elif bc and tc and ltp < bc and entry > bc:
                new_sl = tc + buffer
                if new_sl < current_sl:
                    self.position["stop_loss"] = round(new_sl, 2)
                    risk_reduction = new_sl - entry
                    original_risk = current_sl - entry
                    reduction_pct = (original_risk - risk_reduction) / original_risk * 100 if original_risk > 0 else 0
                    self.log(
                        f"[CPR-SL] ✅ SHORT SL lowered to TC: {new_sl:.2f} (price below BC at {bc:.2f}) | "
                        f"Risk reduced by {reduction_pct:.1f}% (from {original_risk:.2f} to {risk_reduction:.2f} points)",
                        False
                    )
                    self._save_state()
                    return
                else:
                    self.log(
                        f"[CPR-SL] SHORT - Below BC but new SL {new_sl:.2f} not lower than current {current_sl:.2f} - skipping",
                        True
                    )

            else:
                self.log(
                    f"[CPR-SL] SHORT - No CPR level adjustments triggered | "
                    f"Price: {ltp:.2f}, BC: {bc}, S1: {s1}, S2: {s2}, S3: {s3}",
                    True
                )

    # ---------- STRATEGY EXECUTION METHODS ----------
    def execute_unified_strategy(self, ltp, all_inds, primary_tf=None):
        """
        Unified strategy execution with:
        - Options integration (OBX)
        - Gap protection (entry/exit)
        - AI CPR signals
        - Multi-timeframe confluence
        - Dynamic stop loss
        """
        # ==========================================
        # CRITICAL VALIDATION - Fail Fast
        # ==========================================
        if primary_tf is None:
            primary_tf = tf_selected

        if ltp is None or ltp <= 0:
            self.log(f"[UNIFIED] Invalid LTP: {ltp} - aborting cycle", False)
            return

        # ==========================================
        # OPTIONS INTEGRATION - Check OBX First
        # ==========================================
        is_option = getattr(self, "is_option_symbol", False)

        if is_option:
            self.log(f"[OPTIONS] 📊 Option detected: {self.symbol}", True)

            try:
                # Run OBX-specific analysis (Delta, IV, OI flow)
                if getattr(self, "option_engine", None):
                    self.log("[OBX] Running option-specific analysis...", True)
                    self.option_engine.run_tick(self.symbol, lot_qty=self.lot)

                    # Check if OBX managed the position
                    if self.option_engine.position:
                        self.log("[OBX] Position managed by OBX engine - checking gap exits only", True)
                        # Still check gap fail exits for OBX positions
                        current_pos = self.position.get("type", "FLAT")
                        if current_pos != "FLAT":
                            self._handle_gap_exit_only(ltp, all_inds)
                        return  # OBX is handling everything

            except Exception as _e_obx:
                self.log(f"[OBX] Analysis error: {_e_obx}", False)

            # 🔥 If OBX didn't take position, fall through to unified strategy
            self.log("[OPTIONS] OBX passed - continuing to unified strategy with gap protection", True)

        if not all_inds or not isinstance(all_inds, dict):
            self.log("[UNIFIED] Invalid indicators structure - aborting cycle", False)
            return

        self.last_known_ltp = ltp
        self.last_known_inds = all_inds
        self.last_known_primary_tf = primary_tf

        # ==========================================
        # INITIALIZE & VALIDATE
        # ==========================================
        current_pos = self.position.get("type", "FLAT")

        def _tf(tf):
            return self._norm_tf(all_inds, str(tf))

        inds = _tf(primary_tf)

        # Validate indicators exist
        if not isinstance(inds, dict) or not inds.get("timestamp"):
            self.log(f"[UNIFIED] No valid indicators for TF={primary_tf}.", True)
            # Still manage existing positions
            if current_pos != "FLAT":
                self.log("[UNIFIED] Managing existing position with stale data", True)
                try:
                    atr_here = self._get_atr_with_fallback(inds, ltp) if isinstance(inds, dict) else None
                    if atr_here:
                        self._update_trailing_and_breakeven(ltp, atr_here)
                        if isinstance(inds, dict):
                            cpr_analysis = inds.get("cpr_analysis", {})
                            pivot_data = cpr_analysis.get("cpr_levels", {}) if cpr_analysis else {}
                            if pivot_data and "TC" in pivot_data:
                                self._update_dynamic_cpr_stop_loss(ltp, atr_here, pivot_data)
                    if isinstance(inds, dict):
                        self._check_trailing_profit(ltp, inds)
                except Exception as e:
                    self.log(f"[UNIFIED] Error managing position with stale data: {e}", False)
            return

        bar_key = f"{primary_tf}:{inds['timestamp']}"

        # ==========================================
        # GAP PROTECTION - Initialize State
        # ==========================================
        try:
            cpr_analysis = inds.get("cpr_analysis", {}) if isinstance(inds, dict) else {}
            pivot_data = cpr_analysis.get("cpr_levels", {}) if isinstance(cpr_analysis, dict) else {}

            # 🔥 CRITICAL: Fetch DAILY OHLC for gap analysis (not 5m)
            ohlc_gap = None
            try:
                if hasattr(self, 'bot') and callable(getattr(self.bot, 'fetch_ohlc', None)):
                    self.log("[GAP] Fetching daily OHLC for gap analysis", True)
                    ohlc_gap = self.bot.fetch_ohlc(self.symbol, "D", 5)  # 5 days of daily data
                    if ohlc_gap is not None and not ohlc_gap.empty:
                        self.log(f"[GAP] Fetched {len(ohlc_gap)} daily candles", True)
            except Exception as _e_fetch_gap:
                self.log(f"[GAP] OHLC fetch err: {_e_fetch_gap}", True)

            # Parse current timestamp
            try:
                import pandas as _pd_gap
                ts_now = _pd_gap.to_datetime(inds.get('timestamp')).to_pydatetime()
            except Exception:
                ts_now = dt.datetime.now(IST)

            # Reset gap state on new session
            self.gap.reset_if_new_session(ts_now, ohlc_gap, pivot_data)
            self.gap.update_opening_range(ts_now, ohlc_gap)

        except Exception as _e_gmk:
            self.log(f"[GAP] setup error: {_e_gmk}", False)

        # ==========================================
        # GAP FAIL EXIT - Check Before Strategy
        # ==========================================
        if current_pos in ('BUY', 'SELL'):
            try:
                side_lbl = 'LONG' if current_pos == 'BUY' else 'SHORT'
                try:
                    close_px = float(inds.get('close', ltp))
                except Exception:
                    close_px = ltp

                do_exit, msg = self.gap.gap_fail_exit(side_lbl, close_px)
                if do_exit:
                    self.log(f"[GAP-EXIT] {msg} - Closing position", False)
                    if self._process_exit(f"GapFail: {msg}", ltp):
                        self._save_state()
                        return  # Exit processed, stop here
            except Exception as _e_gfail:
                self.log(f"[GAP] fail-exit check err: {_e_gfail}", False)

        # ==========================================
        # CRITICAL GUARDS (HIGHEST PRIORITY)
        # ==========================================

        # Block re-entry on same bar as exit
        if current_pos == "FLAT":
            last_exit_bar = self.position.get("_last_action_bar")
            if last_exit_bar == bar_key:
                self.log(f"[RE-ENTRY BLOCK] Just exited on bar {bar_key} - must wait for next bar", False)
                return

        # Check if we already acted on this bar
        if bar_key == self.position.get("_last_bar_key"):
            if current_pos != "FLAT":
                atr_here = self._get_atr_with_fallback(inds, ltp)
                if atr_here:
                    self._update_trailing_and_breakeven(ltp, atr_here)
                    cpr_analysis = inds.get("cpr_analysis", {})
                    pivot_data = cpr_analysis.get("cpr_levels", {}) if cpr_analysis else {}
                    if pivot_data and "TC" in pivot_data:
                        self._update_dynamic_cpr_stop_loss(ltp, atr_here, pivot_data)
                self._check_trailing_profit(ltp, inds)
            return

        # Update bar tracking
        if bar_key != self.position.get("_last_bar_key"):
            self.position["_exits_this_bar"] = 0
        self.position["_last_bar_key"] = bar_key

        # Deduplication guards
        if self.DEDUPE_ONE_ENTRY_PER_BAR and self.position.get("_last_entry_attempt_bar") == bar_key:
            self.log("[UNIFIED] Entry already attempted this bar - skipping", True)
            return

        # Cooldown check
        if self._cooldown_active():
            self.log("[UNIFIED] Cooldown active - skipping new entries", True)
            if current_pos != "FLAT":
                atr_here = self._get_atr_with_fallback(inds, ltp)
                if atr_here:
                    self._update_trailing_and_breakeven(ltp, atr_here)
                    cpr_analysis = inds.get("cpr_analysis", {})
                    pivot_data = cpr_analysis.get("cpr_levels", {}) if cpr_analysis else {}
                    if pivot_data and "TC" in pivot_data:
                        self._update_dynamic_cpr_stop_loss(ltp, atr_here, pivot_data)
                self._check_trailing_profit(ltp, inds)
            return

        # Re-entry cooldown
        if self.position.get("_skip_entry_until_bar") == bar_key:
            self.log("[UNIFIED] Re-entry cooldown active - skipping", True)
            return

        # ==========================================
        # DETECT TREND (Enhanced with Error Handling)
        # ==========================================
        detected_trend = None
        try:
            df_5 = None
            if hasattr(self, 'bot'):
                try:
                    df_5 = self.bot.fetch_ohlc(self.symbol, "5", 30)
                except Exception as e:
                    self.log(f"[TREND-DETECT] Failed to fetch OHLC: {e}", True)

            if df_5 is not None and isinstance(df_5, pd.DataFrame) and not df_5.empty:
                df_5 = df_5.copy()
                df_5.columns = [c.lower() for c in df_5.columns]
                if "close" in df_5.columns and len(df_5) > 10:
                    i = len(df_5) - 1
                    try:
                        detected_trend = self.detect_trend(df=df_5, i=i)
                        self.log(f"[TREND-DETECT] Result: {detected_trend}", True)
                    except Exception as e:
                        self.log(f"[TREND-DETECT] Detection error: {e}", True)
                        detected_trend = None
        except Exception as e:
            self.log(f"[TREND-DETECT] Critical error: {e}", True)
            detected_trend = None

        # Inject trend into indicators for AI
        if isinstance(inds, dict):
            inds["trend"] = detected_trend

        # 🔥 CONSOLIDATION BLOCKER (Highest Priority)
        if detected_trend and "consol" in str(detected_trend).lower():
            self.log(
                f"[CONSOLIDATION-BLOCK] Market in consolidation - no new entries allowed",
                False
            )
            # Still manage existing positions
            if current_pos != "FLAT":
                self.log("[CONSOLIDATION] Managing existing position during consolidation", True)
                atr_here = self._get_atr_with_fallback(inds, ltp)
                if atr_here:
                    self._update_trailing_and_breakeven(ltp, atr_here)
                    cpr_analysis = inds.get("cpr_analysis", {})
                    pivot_data = cpr_analysis.get("cpr_levels", {}) if cpr_analysis else {}
                    if pivot_data and "TC" in pivot_data:
                        self._update_dynamic_cpr_stop_loss(ltp, atr_here, pivot_data)
                self._check_trailing_profit(ltp, inds)
            return

        # ==========================================
        # SIGNAL COLLECTION FROM ALL SOURCES
        # ==========================================
        signals = {
            "trend": None,
            "ai_cpr": None,
            "cpr_strategy": None,
            "macd": None,
            "ema_cross": None,
            "price_action_trend": None
        }

        confidences = {
            "trend": 0.0,
            "ai_cpr": 0.0,
            "cpr_strategy": 0.0,
            "macd": 0.0,
            "ema_cross": 0.0,
            "price_action_trend": 0.0
        }

        # Extract indicators with validation
        e5 = self._f(inds.get("ema_5"))
        e9 = self._f(inds.get("ema_9"))
        e21 = self._f(inds.get("ema_21"))
        macd_color = inds.get("macd_color")
        adx_val = self._f(inds.get("adx"))
        bb_bandwidth = self._f(inds.get("bb_bandwidth"))

        # Volatility filter
        is_choppy = bb_bandwidth is not None and bb_bandwidth < 0.005
        if is_choppy:
            self.log(
                f"[UNIFIED] Choppy market detected (BBW: {bb_bandwidth:.4f}) - blocking entries",
                False
            )
            if current_pos != "FLAT":
                self.log("[CHOPPY] Managing existing position in choppy market", True)
                atr_here = self._get_atr_with_fallback(inds, ltp)
                if atr_here:
                    self._update_trailing_and_breakeven(ltp, atr_here)
                    cpr_analysis = inds.get("cpr_analysis", {})
                    pivot_data = cpr_analysis.get("cpr_levels", {}) if cpr_analysis else {}
                    if pivot_data and "TC" in pivot_data:
                        self._update_dynamic_cpr_stop_loss(ltp, atr_here, pivot_data)
                self._check_trailing_profit(ltp, inds)
            return

        # 1️⃣ TREND-BASED SIGNAL (EMA Cross)
        if e9 is not None and e21 is not None:
            if e9 > e21:
                signals["ema_cross"] = "BUY"
                confidences["ema_cross"] = 0.6
            elif e9 < e21:
                signals["ema_cross"] = "SELL"
                confidences["ema_cross"] = 0.6

        # 2️⃣ MACD SIGNAL
        adx_strong = adx_val is not None and adx_val > 20
        if macd_color and adx_strong:
            if macd_color == "Dark Green":
                signals["macd"] = "BUY"
                confidences["macd"] = 0.7
            elif macd_color == "Dark Red":
                signals["macd"] = "SELL"
                confidences["macd"] = 0.7

        # 3️⃣ COMBINED TREND SIGNAL (EMA + MACD)
        if signals["ema_cross"] == "BUY" and signals["macd"] == "BUY":
            signals["trend"] = "BUY"
            confidences["trend"] = 0.8
        elif signals["ema_cross"] == "SELL" and signals["macd"] == "SELL":
            signals["trend"] = "SELL"
            confidences["trend"] = 0.8

        # 4️⃣ PRICE ACTION TREND SIGNAL
        if detected_trend:
            trend_lower = str(detected_trend).lower()
            if "up" in trend_lower or trend_lower == "uptrend":
                signals["price_action_trend"] = "BUY"
                confidences["price_action_trend"] = 0.65
                self.log(f"[PRICE-ACTION] Uptrend detected via candle structure", True)
            elif "down" in trend_lower or trend_lower == "downtrend":
                signals["price_action_trend"] = "SELL"
                confidences["price_action_trend"] = 0.65
                self.log(f"[PRICE-ACTION] Downtrend detected via candle structure", True)

        # ==========================================
        # 5️⃣ AI CPR SIGNAL - REAL-TIME PREDICTION
        # ==========================================
        ai_prediction = None
        ai_confidence_raw = 0.0
        ai_distribution = None

        try:
            pivot_data = inds.get("cpr_analysis", {})

            if not pivot_data or not isinstance(pivot_data, dict) or "TC" not in pivot_data:
                try:
                    pivot_json_path = self.state_path.replace("om_state", "pivot")
                    loaded_pivot = robust_load_json(pivot_json_path, self.log, default={})
                    pivot_data = loaded_pivot.get(self.symbol, {}) if isinstance(loaded_pivot, dict) else {}
                except Exception as e:
                    self.log(f"[AI-CPR] Failed to load pivot data: {e}", True)
                    pivot_data = {}

            if pivot_data and "TC" in pivot_data and "BC" in pivot_data:
                ai_label, ai_conf, ai_dist, _ = self.ai_predictor.predict(
                    indicators=inds,
                    pivot_data=pivot_data,
                    feature_builder=_build_ai_cpr_features
                )

                ai_confidence_raw = float(ai_conf) if ai_conf is not None else 0.0
                ai_distribution = ai_dist

                self.log(
                    f"[AI-CPR] ========== AI PREDICTION DETAILS ==========",
                    False
                )
                self.log(
                    f"[AI-CPR] Label: {ai_label} | Confidence: {ai_confidence_raw:.4f} ({ai_confidence_raw * 100:.2f}%)",
                    False
                )
                self.log(
                    f"[AI-CPR] Threshold: {self.AI_MIN_CONF:.4f} | Status: {'✅ ACCEPTED' if ai_confidence_raw >= self.AI_MIN_CONF else '❌ REJECTED'}",
                    False
                )
                if ai_distribution:
                    self.log(f"[AI-CPR] Distribution: {ai_distribution}", False)
                self.log(f"[AI-CPR] ==========================================", False)

                if ai_label and ai_confidence_raw >= self.AI_MIN_CONF:
                    ai_label_upper = str(ai_label).upper()

                    if any(keyword in ai_label_upper for keyword in ["BUY", "BULLISH", "LONG", "UP"]):
                        signals["ai_cpr"] = "BUY"
                        confidences["ai_cpr"] = ai_confidence_raw
                        self.log(f"[AI-CPR] ✅ BUY signal accepted (conf: {ai_confidence_raw:.3f})", True)

                    elif any(keyword in ai_label_upper for keyword in ["SELL", "BEARISH", "SHORT", "DOWN"]):
                        signals["ai_cpr"] = "SELL"
                        confidences["ai_cpr"] = ai_confidence_raw
                        self.log(f"[AI-CPR] ✅ SELL signal accepted (conf: {ai_confidence_raw:.3f})", True)

                    else:
                        self.log(f"[AI-CPR] ⚠️ Neutral/Hold signal: {ai_label}", True)
                else:
                    conf_msg = f"{ai_confidence_raw:.3f}" if ai_conf is not None else "N/A"
                    self.log(
                        f"[AI-CPR] ❌ Signal rejected: confidence {conf_msg} < threshold {self.AI_MIN_CONF}",
                        True
                    )
            else:
                self.log("[AI-CPR] ⚠️ Pivot data incomplete - skipping AI prediction", True)

        except Exception as e:
            self.log(f"[AI-CPR] ⚠️ Prediction error: {e}", False)
            import traceback
            self.log(f"[AI-CPR] Traceback: {traceback.format_exc()}", True)

        # 6️⃣ CPR STRATEGY SIGNAL
        cpr_analysis = inds.get("cpr_analysis", {})

        if not cpr_analysis or not isinstance(cpr_analysis, dict):
            self.log("[CPR-STRATEGY] CPR analysis missing - signal skipped", True)
        else:
            cpr_levels = cpr_analysis.get("cpr_levels", {})
            cpr_valid = bool(
                cpr_levels and
                isinstance(cpr_levels, dict) and
                cpr_levels.get("TC") and
                cpr_levels.get("BC") and
                cpr_levels.get("R1") and
                cpr_levels.get("S1")
            )

            if not cpr_valid:
                self.log("[CPR-STRATEGY] CPR levels incomplete - signal skipped", True)
            else:
                cpr_trade_signal = cpr_analysis.get("trade_strategy", "None")

                if cpr_trade_signal == "Buy":
                    signals["cpr_strategy"] = "BUY"
                    confidences["cpr_strategy"] = 0.7
                elif cpr_trade_signal == "Sell":
                    signals["cpr_strategy"] = "SELL"
                    confidences["cpr_strategy"] = 0.7

        # ==========================================
        # INTELLIGENT VOTING & CONFLUENCE
        # ==========================================
        buy_votes = []
        sell_votes = []

        for source, signal in signals.items():
            if signal == "BUY":
                buy_votes.append((source, confidences[source]))
            elif signal == "SELL":
                sell_votes.append((source, confidences[source]))

        buy_count = len(buy_votes)
        sell_count = len(sell_votes)

        buy_score = sum(conf for _, conf in buy_votes)
        sell_score = sum(conf for _, conf in sell_votes)

        self.log(
            f"[UNIFIED] Signal Analysis:\n"
            f"  BUY: {buy_count} votes (score: {buy_score:.2f}) - {[f'{s}({c:.2f})' for s, c in buy_votes]}\n"
            f"  SELL: {sell_count} votes (score: {sell_score:.2f}) - {[f'{s}({c:.2f})' for s, c in sell_votes]}",
            True
        )

        # ==========================================
        # DECISION LOGIC WITH HARDENED THRESHOLDS
        # ==========================================
        final_signal = None
        reason = ""

        ai_in_buy = any(s == "ai_cpr" for s, _ in buy_votes)
        ai_in_sell = any(s == "ai_cpr" for s, _ in sell_votes)

        # 🔥 PRIORITY 1: High confidence AI signal (≥0.75)
        if ai_in_buy and confidences["ai_cpr"] >= 0.75:
            opposing_signals = sell_count
            supporting_signals = buy_count - 1

            if supporting_signals >= 1:
                final_signal = "BUY"
                reason = f"HIGH CONFIDENCE AI BUY ({confidences['ai_cpr']:.2f}) + {supporting_signals} support"
            elif opposing_signals <= 2:
                final_signal = "BUY"
                reason = f"HIGH CONFIDENCE AI BUY ({confidences['ai_cpr']:.2f}) solo (weak opposition: {opposing_signals})"
            else:
                self.log(
                    f"[AI-BLOCK] AI BUY confidence {confidences['ai_cpr']:.2f} but {opposing_signals} strong opposing signals - too risky",
                    False
                )

        elif ai_in_sell and confidences["ai_cpr"] >= 0.75:
            opposing_signals = buy_count
            supporting_signals = sell_count - 1

            if supporting_signals >= 1:
                final_signal = "SELL"
                reason = f"HIGH CONFIDENCE AI SELL ({confidences['ai_cpr']:.2f}) + {supporting_signals} support"
            elif opposing_signals <= 2:
                final_signal = "SELL"
                reason = f"HIGH CONFIDENCE AI SELL ({confidences['ai_cpr']:.2f}) solo (weak opposition: {opposing_signals})"
            else:
                self.log(
                    f"[AI-BLOCK] AI SELL confidence {confidences['ai_cpr']:.2f} but {opposing_signals} strong opposing signals - too risky",
                    False
                )

        # 🔥 PRIORITY 2: Strong confluence (3+ votes)
        elif buy_count >= 3:
            final_signal = "BUY"
            ai_part = f" + AI({confidences['ai_cpr']:.2f})" if ai_in_buy else ""
            reason = f"Strong BUY confluence ({buy_count}/6 votes, score: {buy_score:.2f}){ai_part}"

        elif sell_count >= 3:
            final_signal = "SELL"
            ai_part = f" + AI({confidences['ai_cpr']:.2f})" if ai_in_sell else ""
            reason = f"Strong SELL confluence ({sell_count}/6 votes, score: {sell_score:.2f}){ai_part}"

        # 🔥 PRIORITY 3: Medium confluence (2 votes)
        elif buy_count == 2:
            if ai_in_buy and confidences["ai_cpr"] >= 0.65 and buy_score >= 1.3:
                final_signal = "BUY"
                reason = f"Medium BUY confluence (2/6 votes, AI: {confidences['ai_cpr']:.2f}, score: {buy_score:.2f})"
            elif buy_score >= 1.5:
                final_signal = "BUY"
                reason = f"Medium BUY confluence (2/6 votes, high score: {buy_score:.2f})"
            else:
                self.log(
                    f"[ENTRY-BLOCK] 2 BUY votes but weak (score: {buy_score:.2f}, AI: {confidences.get('ai_cpr', 0):.2f}) - waiting for stronger setup",
                    True
                )

        elif sell_count == 2:
            if ai_in_sell and confidences["ai_cpr"] >= 0.65 and sell_score >= 1.3:
                final_signal = "SELL"
                reason = f"Medium SELL confluence (2/6 votes, AI: {confidences['ai_cpr']:.2f}, score: {sell_score:.2f})"
            elif sell_score >= 1.5:
                final_signal = "SELL"
                reason = f"Medium SELL confluence (2/6 votes, high score: {sell_score:.2f})"
            else:
                self.log(
                    f"[ENTRY-BLOCK] 2 SELL votes but weak (score: {sell_score:.2f}, AI: {confidences.get('ai_cpr', 0):.2f}) - waiting for stronger setup",
                    True
                )

        # ==========================================
        # EXECUTE ENTRY WITH GAP CHECK
        # ==========================================
        if final_signal and current_pos == "FLAT":
            # 🔥 GAP ENTRY BLOCK CHECK
            try:
                ema5_val = self._f(inds.get("ema_5"))
                ema21_val = self._f(inds.get("ema_21"))
                side_word = "Buy" if final_signal == "BUY" else "Sell"

                blocked, why, sl_override = self.gap.block_entry(
                    side_word,
                    dt.datetime.now(IST),
                    ltp,
                    ema5_val,
                    ema21_val
                )

                if blocked:
                    self.log(f"[GAP-BLOCK] Entry blocked: {why}", False)
                    return  # Don't proceed with entry

                if sl_override is not None:
                    self.dynamic_sl_override = sl_override
                    self.log(f"[GAP] Dynamic SL override: {sl_override:.2f}", True)

            except Exception as _ge:
                self.log(f"[GAP] guard error: {_ge}", True)

            # Proceed with entry
            atr_here = self._get_atr_with_fallback(inds, ltp)
            if atr_here:
                sources = [f"{s}({c:.2f})" for s, c in (buy_votes if final_signal == "BUY" else sell_votes)]
                full_reason = f"{reason} | Sources: {', '.join(sources)}"

                if self._process_entry(final_signal, full_reason, ltp, atr_here, bar_key=bar_key, indsP=inds):
                    self.position["_last_action_bar"] = bar_key
                    self.position["ai_entry_confidence"] = confidences.get("ai_cpr", 0.0)
                    self.position["ai_distribution"] = ai_distribution
                    self._set_cooldown(self.FLIP_COOLDOWN_BARS)
                    self._save_state()
                    self.log(
                        f"[UNIFIED] ✅ Entry executed: {final_signal} | AI Confidence: {confidences.get('ai_cpr', 'N/A')}",
                        False
                    )
                    return

        # ==========================================
        # EXIT LOGIC - MULTIPLE LAYERS
        # ==========================================
        reversal_at_resistance = False
        candle_patterns = {}
        pivot_interactions = {}

        if cpr_analysis and isinstance(cpr_analysis, dict):
            reversal_at_resistance = cpr_analysis.get("reversal_at_r2_r3", False)
            candle_patterns = cpr_analysis.get("candle_patterns", {})
            pivot_interactions = cpr_analysis.get("pivot_interactions", {})

            if not isinstance(candle_patterns, dict):
                candle_patterns = {}
            if not isinstance(pivot_interactions, dict):
                pivot_interactions = {}

        # 🔥 LAYER 1: CPR REVERSAL AT MAJOR LEVELS
        if current_pos == "BUY" and reversal_at_resistance:
            bearish_signals = []
            if candle_patterns.get("big_bear_takeout"):
                bearish_signals.append("Big Bear Take Out")
            if candle_patterns.get("fake_bull"):
                bearish_signals.append("Fake Bull")
            if candle_patterns.get("bear_retracement"):
                bearish_signals.append("Bear Retracement")

            if bearish_signals:
                pattern_str = ", ".join(bearish_signals)
                reason = f"EXIT LONG - Bearish reversal at R2/R3 resistance ({pattern_str})"
                self.log(f"[CPR-REVERSAL] {reason}", False)

                if self._process_exit(reason, ltp):
                    self._set_cooldown(1)
                    self._save_state()
                    return

        if current_pos == "SELL":
            # Check for bullish reversal at S2/S3 support
            at_s2 = pivot_interactions.get("at_s2", False)
            at_s3 = pivot_interactions.get("at_s3", False)
            tested_s2 = pivot_interactions.get("tested_s2", False)
            tested_s3 = pivot_interactions.get("tested_s3", False)

            bullish_signals = []
            if candle_patterns.get("big_bull_takeout"):
                bullish_signals.append("Big Bull Take Out")
            if candle_patterns.get("fake_bear"):
                bullish_signals.append("Fake Bear")
            if candle_patterns.get("bull_retracement"):
                bullish_signals.append("Bull Retracement")

            if (at_s2 or at_s3 or tested_s2 or tested_s3) and bullish_signals:
                pattern_str = ", ".join(bullish_signals)
                level = "S2" if (at_s2 or tested_s2) else "S3"
                reason = f"EXIT SHORT - Bullish reversal at {level} support ({pattern_str})"
                self.log(f"[CPR-REVERSAL] {reason}", False)

                if self._process_exit(reason, ltp):
                    self._set_cooldown(1)
                    self._save_state()
                    return

        # 🔥 LAYER 2: HIGH CONFIDENCE AI REVERSAL
        if current_pos == "BUY":
            if ai_in_sell and confidences["ai_cpr"] >= 0.70:
                reason = f"EXIT LONG - High confidence AI reversal (SELL: {confidences['ai_cpr']:.2f})"
                self.log(f"[AI-EXIT] {reason} | Distribution: {ai_distribution}", False)

                if self._process_exit(reason, ltp):
                    self._set_cooldown(1)
                    self._save_state()
                    return

        elif current_pos == "SELL":
            if ai_in_buy and confidences["ai_cpr"] >= 0.70:
                reason = f"EXIT SHORT - High confidence AI reversal (BUY: {confidences['ai_cpr']:.2f})"
                self.log(f"[AI-EXIT] {reason} | Distribution: {ai_distribution}", False)

                if self._process_exit(reason, ltp):
                    self._set_cooldown(1)
                    self._save_state()
                    return

        # 🔥 LAYER 3: MULTIPLE SIGNAL REVERSAL
        if current_pos == "BUY":
            if sell_count >= 3 or (sell_count >= 2 and sell_score > 1.4):
                ai_part = f" + AI({confidences['ai_cpr']:.2f})" if ai_in_sell else ""
                reason = f"EXIT LONG - Strong reversal ({sell_count} SELL votes, score: {sell_score:.2f}){ai_part}"
                self.log(f"[MULTI-EXIT] {reason}", False)

                if self._process_exit(reason, ltp):
                    self._set_cooldown(1)
                    self._save_state()
                    return

        elif current_pos == "SELL":
            if buy_count >= 3 or (buy_count >= 2 and buy_score > 1.4):
                ai_part = f" + AI({confidences['ai_cpr']:.2f})" if ai_in_buy else ""
                reason = f"EXIT SHORT - Strong reversal ({buy_count} BUY votes, score: {buy_score:.2f}){ai_part}"
                self.log(f"[MULTI-EXIT] {reason}", False)

                if self._process_exit(reason, ltp):
                    self._set_cooldown(1)
                    self._save_state()
                    return

        # ==========================================
        # MAINTAIN EXISTING POSITION
        # ==========================================

        if current_pos != "FLAT":
            atr_here = self._get_atr_with_fallback(inds, ltp)
            if atr_here:
                self._update_trailing_and_breakeven(ltp, atr_here)
                # 🔥 Dynamic CPR-based stop loss
                cpr_analysis = inds.get("cpr_analysis", {})
                pivot_data = cpr_analysis.get("cpr_levels", {}) if cpr_analysis else {}  # 🔥 EXTRACT nested cpr_levels
                if pivot_data and "TC" in pivot_data:  # 🔥 VALIDATE before calling
                    self._update_dynamic_cpr_stop_loss(ltp, atr_here, pivot_data)
            self._check_trailing_profit(ltp, inds)
        self._save_state()

    def _handle_gap_exit_only(self, ltp, inds):

        current_pos = self.position.get("type", "FLAT")
        if current_pos in ('BUY', 'SELL'):
            try:
                side_lbl = 'LONG' if current_pos == 'BUY' else 'SHORT'
                close_px = ltp

                do_exit, msg = self.gap.gap_fail_exit(side_lbl, close_px)
                if do_exit:
                    self.log(f"[GAP-EXIT-EMERGENCY] {msg} - Emergency close", False)
                    if self._process_exit(f"GapFail (emergency): {msg}", ltp):
                        self._save_state()
            except Exception as _e:
                self.log(f"[GAP] emergency exit err: {_e}", False)

    def detect_trend(self, df, i=-1):

        try:
            if len(df) < 20:
                return None

            # Normalize column names
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]

            if 'close' not in df:
                return None

            # Calculate EMAs
            ema_fast = df['close'].ewm(span=5, adjust=False).mean()
            ema_slow = df['close'].ewm(span=15, adjust=False).mean()

            last_fast = ema_fast.iloc[i]
            last_slow = ema_slow.iloc[i]

            # Get recent candles
            recent = df.iloc[-10:].copy()

            # 1. EMA Alignment Check
            ema_bullish = last_fast > last_slow * 1.001  # At least 0.1% above
            ema_bearish = last_fast < last_slow * 0.999

            # 2. Price Structure Check (Higher Highs/Lower Lows)
            if 'high' in df.columns and 'low' in df.columns:
                highs = recent['high'].values
                lows = recent['low'].values

                # Check for higher highs (uptrend)
                higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
                higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])

                # Check for lower lows (downtrend)
                lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i - 1])
                lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i - 1])

                uptrend_structure = (higher_highs >= 6 and higher_lows >= 6)
                downtrend_structure = (lower_highs >= 6 and lower_lows >= 6)
            else:
                uptrend_structure = False
                downtrend_structure = False

            # 3. Slope Check
            closes = recent['close'].values
            slope = (closes[-1] - closes[0]) / len(closes)
            avg_price = closes.mean()
            slope_pct = (slope / avg_price) * 100 if avg_price > 0 else 0

            strong_up_slope = slope_pct > 0.05  # > 0.05% per candle
            strong_down_slope = slope_pct < -0.05

            # 4. Volatility/Consolidation Check
            if 'high' in df.columns and 'low' in df.columns:
                ranges = recent['high'] - recent['low']
                avg_range = ranges.mean()
                current_range = recent['high'].iloc[-1] - recent['low'].iloc[-1]

                consolidating = (
                        current_range < avg_range * 0.5 and
                        recent['close'].std() < recent['close'].mean() * 0.01  # ← Add price stability check
                )
            else:
                consolidating = False

            # 5. Volume Confirmation (if available)
            volume_confirmed = False
            if 'volume' in df.columns:
                recent_vol = recent['volume'].values
                avg_vol = recent_vol[:-3].mean() if len(recent_vol) > 3 else 0
                current_vol = recent_vol[-1]

                # Strong move with volume > 120% of average
                volume_confirmed = current_vol > avg_vol * 1.2

            # DECISION LOGIC with scoring
            uptrend_score = 0
            downtrend_score = 0

            if ema_bullish:
                uptrend_score += 2
            if uptrend_structure:
                uptrend_score += 3
            if strong_up_slope:
                uptrend_score += 2
            if volume_confirmed and slope_pct > 0:
                uptrend_score += 1

            if ema_bearish:
                downtrend_score += 2
            if downtrend_structure:
                downtrend_score += 3
            if strong_down_slope:
                downtrend_score += 2
            if volume_confirmed and slope_pct < 0:
                downtrend_score += 1

            # Return based on scores
            if consolidating:
                return "Consolidation"
            elif uptrend_score >= 5:
                return "Uptrend"
            elif downtrend_score >= 5:
                return "Downtrend"
            else:
                return None  # Unclear/transitioning

        except Exception as e:
            self.log(f"[ERROR] Enhanced detect_trend failed: {e}", True)
            return None

    # --- Helper: Cooldown management ---
    def _cooldown_active(self):
        """
        Returns True if cooldown is active (still waiting before next trade)
        """
        try:
            cd = int(self.position.get("_cooldown_bars", 0) or 0)
        except Exception:
            cd = 0

        if cd > 0:
            # reduce cooldown on each cycle
            self.position["_cooldown_bars"] = cd - 1
            self.log(f"[COOLDOWN] Waiting for {cd - 1} more bars before new trade", True)
            return True
        return False

    def _set_cooldown(self, bars: int = 1):
        """
        Activates cooldown after a trade to prevent overtrading
        """
        self.position["_cooldown_bars"] = bars
        self.log(f"[COOLDOWN] Set for {bars} bars", True)

    def _ai_place_order(self, ai_signal, reason="AI-CPR Signal"):
        """
        Safely execute AI-based orders with position awareness and cooldown checks.
        Handles both dict and string AI signals, and ensures correct order direction.
        """
        try:
            self.log("DEBUG: Entered Order Manager _ai_place_order", False)

            current_side = self.position.get("type")
            cooldown_active = self._cooldown_active()

            # --- Normalize AI Signal (handles dict or string) ---
            if isinstance(ai_signal, dict):
                ai_signal = ai_signal.get("action")

            if not ai_signal:
                self.log("[INFO] Empty AI signal — skipping order", True)
                return

            ai_signal = str(ai_signal).capitalize().strip()
            if ai_signal not in ("Bullish", "Bearish"):
                self.log(f"[INFO] Invalid AI signal ({ai_signal}) — skipping order", True)
                return

            desired_side = "BUY" if ai_signal == "Bullish" else "SELL"

            # --- Cooldown enforcement ---
            if cooldown_active:
                self.log(f"[COOLDOWN] Skipping {desired_side} due to cooldown", True)
                return

            # --- Avoid duplicate entries ---
            if current_side == desired_side:
                self.log(f"[HOLD] Already in {current_side}, skipping re-entry.", True)
                return

            # --- Handle reversal (BUY → SELL or SELL → BUY) ---
            if current_side and current_side != desired_side:
                self.log(f"[REVERSAL] Closing {current_side} → Opening {desired_side} ({reason})", True)
                try:
                    self.bot.place_order(self.symbol, side="EXIT", qty=self.lot, reason="Reversal Exit")
                except Exception as e:
                    self.log(f"[ERROR] Failed to exit {current_side}: {e}", True)
                    return

                # Place new order
                try:
                    self.bot.place_order(self.symbol, side=desired_side, qty=self.lot, reason=reason)
                    self.position["side"] = desired_side
                    self._set_cooldown(1)
                except Exception as e:
                    self.log(f"[ERROR] Failed to place {desired_side} order: {e}", True)
                return

            # --- New entry if flat (no current position) ---
            if not current_side:
                self.log(f"[ENTRY] Placing {desired_side} order for {self.symbol} ({reason})", True)
                self.bot.place_order(self.symbol, side=desired_side, qty=self.lot, reason=reason)
                self.position["side"] = desired_side
                self._set_cooldown(1)
                return

        except Exception as e:
            self.log(f"[ERROR] _ai_place_order() failed: {e}", True)

    # ---------- AI CPR STATE MANAGEMENT ----------
    def get_ai_state(self):
        """Get current AI CPR state for monitoring"""
        return {
            "last_ai_action": self.last_ai_action,
            "last_ai_confidence": self.last_ai_confidence,
            "ai_enabled": self.AI_CPR_ENABLED,
            "ai_gate_trades": self.AI_GATE_TRADES,
            "ai_min_confidence": self.AI_MIN_CONF
        }

    def update_ai_config(self, enabled=None, min_conf=None, gate_trades=None):
        """Update AI CPR configuration dynamically"""
        if enabled is not None:
            self.AI_CPR_ENABLED = enabled
            self.log(f"[AI-CPR] Strategy {'enabled' if enabled else 'disabled'}", False)

        if min_conf is not None:
            self.AI_MIN_CONF = min_conf
            self.log(f"[AI-CPR] Minimum confidence set to {min_conf}", False)

        if gate_trades is not None:
            self.AI_GATE_TRADES = gate_trades
            self.log(f"[AI-CPR] Trade gating {'enabled' if gate_trades else 'disabled'}", False)

    # ---------- CSV LOGGING METHODS ----------
    def _ensure_trade_csv(self):
        """Ensure trade CSV has AI-specific columns"""
        if not self.trades_csv:
            return

        if not os.path.exists(self.trades_csv):
            fields = [
                "trade_id","symbol","side","event",
                "entry_time","exit_time","hold_seconds",
                "entry_ltp","exit_ltp","ltp_diff",
                "reason","order_id","bar_key",
                "adx","macd_color","ema5","ema9","ema21",
                "ai_confidence","ai_action"
            ]
            with open(self.trades_csv, "w", newline="") as f:
                w = DictWriter(f, fieldnames=fields)
                w.writeheader()

    def _append_trade_csv(self, row: dict):
        """Append trade with AI data"""
        if not self.trades_csv:
            return

        fields = [
            "trade_id","symbol","side","event",
            "entry_time","exit_time","hold_seconds",
            "entry_ltp","exit_ltp","ltp_diff",
            "reason","order_id","bar_key",
            "adx","macd_color","ema5","ema9","ema21",
            "ai_confidence","ai_action"
        ]

        # Ensure all fields are present
        for k in fields:
            row.setdefault(k, "")

        with open(self.trades_csv, "a", newline="") as f:
            w = DictWriter(f, fieldnames=fields)
            w.writerow(row)

    # ---------- DASHBOARD EXPORT ----------
    def export_normalized_dashboard(self, all_inds, out_path, tfs=("1","5","15","30")):
        dash = {}
        for tf in tfs:
            d = self._norm_tf(all_inds, tf)
            if d and any(k in d for k in ("ema_5","ema_9","ema_21","close","ATR","atr")):
                dash[str(tf)] = {"inds": d, "ts": d.get("timestamp")}
        if not dash:
            self.log("[EXPORT] No TFs normalized; likely a raw list or wrong shape was passed.", False)
            return False
        robust_save_json({"Dashboard": dash}, out_path, self.log)
        self.log(f"[EXPORT] Wrote {out_path} with TFs: {', '.join(sorted(dash.keys()))}", False)
        return True



class TradingBot:
    def __init__(self, config_dir="config", run_websocket=True):
        self.IST   = IST
        self.DEBUG = True

        # Trading configuration
        #self.symbols = ["MCX:NATGASMINI25OCTFUT"]  # Your symbol
        #self.symbols = ["MCX:GOLDPETAL25OCTFUT"]
        #self.symbols = ["MCX:NATGASMINI25OCT300PE"]
        self.symbols = ["NSE:LAURUSLABS-EQ"]
        #self.symbols = ["NSE:MAZDOCK-EQ"]
        self.symbol  = self.symbols[0]
        self.symbol_clean = self.symbol.replace(":", "_")

        # Strategy flags - ADD THESE NEW FLAGS
        self.USE_TREND_STRATEGY = True    # Use traditional trend-based strategy
        self.USE_AI_CPR_STRATEGY = True   # Use AI CPR-based strategy
        self.USE_COMBINED_STRATEGY = True # Use both strategies together

        self._setup_paths()
        self._initialize_symbol_files()
        self._load_config(config_dir)

        # Initialize Fyers SDK
        self.fyers_sdk_instance = fyersModel.FyersModel(
            client_id=self.client_id,
            token=self.access_token,
            is_async=False,
            log_path=""
        )

        # Initialize AI Predictor - ENHANCED INITIALIZATION
        self.ai_predictor = CPR_AIPredictor(
            model_path="ai_cpr_model.pkl",  # Specify model path
            logger=logger
        )

        # Websocket caches
        self.websocket_ltp  = {}
        self.websocket_lock = threading.Lock()
        self._last_print_time = {}
        self.fresh_indicators_all_tfs = {}

        # Websocket OHLC accumulation
        self.ohlc_data = {}
        self.csv_data = {}
        self.timeframe_counter = 1
        self.timeframe = 5  # in minutes
        self.ohlc_lock = threading.Lock()
        self._last_minute_seen = None

        if run_websocket:
            self._setup_websocket()
        else:
            self.log_message("WebSocket disabled.", False)

        # Per-symbol order managers & candle tracking
        self.order_managers = {}
        self.last_bar_ts    = {sym: None for sym in self.symbols}
        self.prev_st21      = {sym: None for sym in self.symbols}

        # ENHANCED ORDER MANAGER INITIALIZATION
        for sym in self.symbols:
            fyers_service = FyersService(
                self.fyers_sdk_instance,
                self.data_paths[sym]['raw_api_log'],
                self.log_message,
                self.get_websocket_ltp
            )
            self.order_managers[sym] = OrderManager(
                fyers_service=fyers_service,
                symbol=sym,
                lot_size=1,
                log_fn=self.log_message,
                state_path=self.data_paths[sym]['om_state_path'],
                event_log=self.data_paths[sym]['om_event_log'],
                report_dir=self.data_paths[sym]['trade_report_dir'],
                ai_predictor=self.ai_predictor,  # Pass AI predictor
                bot=self  # Pass bot reference for fetch_ohlc
            )
            # Configure signal modes
            self.order_managers[sym].SIGNAL_MODE = "both"
            # Configure AI CPR settings
            self.order_managers[sym].AI_CPR_ENABLED = self.USE_AI_CPR_STRATEGY
            self.order_managers[sym].AI_MIN_CONF = 0.55
            self.order_managers[sym].AI_GATE_TRADES = True

        self.candle_analyzer      = CandlestickAnalyzer(self)
        self.indicator_calculator = IndicatorCalculator(self)
        self.previous_states = {tf: {} for tf in ['1','5','15','30','60']}

    def run(self, selected_tf="5"):
        """
        Unified main trading loop:
        Combines AI CPR, Trend-based, and Combined strategy execution.
        """
        self.initialize_pivots()
        pivot_data = robust_load_json(self.pivot_json, self.log_message, default={})
        if self.symbol in pivot_data:
            pivot_ts = pivot_data[self.symbol].get("ts", "1970-01-01")
            try:
                pivot_age = pd.to_datetime(pivot_ts) if pivot_ts != "1970-01-01" else pd.Timestamp.min
                if (pd.Timestamp.now(tz=IST) - pivot_age).total_seconds() > 86400:  # 24 hours
                    self.log_message("Pivots are stale (>24h), forcing recalculation", False)
                    self.process_pivots()  # Force recalculate
            except Exception as e:
                self.log_message(f"Error checking pivot age: {e}, forcing recalc", False)
                self.process_pivots()

        timeframes_to_process = ["1", "5", "15", "30"]

        while True:
            now = dt.datetime.now(self.IST)
            fresh_indicators_all_tfs = {}

            # ─────────────────────────── INDICATOR CALCULATION ───────────────────────────
            for sym in self.symbols:
                all_data_json_path = self.data_paths[sym]['all_data_json']
                dashboard_data = robust_load_json(all_data_json_path, self.log_message, default={})
                dashboard_data.setdefault("Dashboard", {})

                # Load pivot data and log for debugging
                pivots = robust_load_json(self.data_paths[sym]['pivot_json'], self.log_message, default={}).get(sym, {})
                self.log_message(f"[DEBUG] Loaded pivots for {sym}: TC={pivots.get('TC', 'Missing')}, BC={pivots.get('BC', 'Missing')}", True)
                fresh_indicators_all_tfs[sym] = {}

                # Fetch OHLC once for candle pattern analysis
                ohlc_df_for_patterns = None
                if self.USE_AI_CPR_STRATEGY:
                    try:
                        ohlc_df_for_patterns = self.fetch_ohlc(sym, "5", 2)  # Fetch 2 days of data for patterns
                        self.log_message(f"[DEBUG] Fetched OHLC for {sym} patterns: {len(ohlc_df_for_patterns) if ohlc_df_for_patterns is not None else 0} rows", True)
                    except Exception as e:
                        self.log_message(f"[WARN] Failed to fetch OHLC for patterns {sym} 5m: {e}", True)

                for tf in timeframes_to_process:
                    self.log_message(f"Calculating indicators for {sym} on {tf}min timeframe...", True)
                    indicators = self.indicator_calculator.calculate_indicators(sym, tf, pivot_data=pivots)
                    fresh_indicators_all_tfs[sym][tf] = indicators

                    # Optional AI-CPR Analysis (Real-time friendly)
                    if ("error" not in indicators and self.USE_AI_CPR_STRATEGY):

                        try:
                            cpr_analysis = analyze_cpr_strategy(indicators, pivots, self.ai_predictor, ohlc_df=ohlc_df_for_patterns)
                            indicators["cpr_analysis"] = cpr_analysis
                            trade_signal = cpr_analysis.get('trade_strategy', 'None')
                            ai_status = ""
                            if cpr_analysis.get('ai_cpr_label'):
                                ai_conf = cpr_analysis.get('ai_confidence', 0.0)
                                ai_status = f" | AI:{cpr_analysis['ai_cpr_label']}({ai_conf:.2f})"
                            self.log_message(
                                f"[AI-CPR] {sym} {tf}m → {trade_signal}{ai_status}",
                                True
                            )
                        except Exception as e:
                            self.log_message(f"[AI-CPR] Analysis failed for {sym} {tf}m: {e}", False)
                            indicators["cpr_analysis"] = {"error": str(e), "trade_strategy": "None"}
                    else:
                        # Ensure cpr_analysis exists even when skipped
                        indicators["cpr_analysis"] = {"trade_strategy": "None", "reason": "Insufficient data"}

                    if "error" in indicators:
                        dashboard_data["Dashboard"][tf] = {
                            "error": indicators.get("error"),
                            "ts": now.isoformat()
                        }
                    else:
                        dashboard_data["Dashboard"][tf] = {
                            "inds":   convert_dict_to_serializable(indicators),
                            "pivots": pivots,
                            "ts": now.isoformat()
                        }

                robust_save_json(dashboard_data, all_data_json_path, self.log_message, debug_only=True)

            # ─────────────────────────── STRATEGY EXECUTION ───────────────────────────
            for sym in self.symbols:
                five_inds = fresh_indicators_all_tfs.get(sym, {}).get("5")

                #self.log_message(f"[DEBUG] 5m indicators for {sym}: {five_inds.keys() if five_inds else 'None'}", False)

                if not five_inds or "error" in five_inds:
                    self.log_message(f"[WARN] Using last known indicators for {sym} due to 5m calc error.", False)
                    five_inds = robust_load_json(self.data_paths[sym]['all_data_json'], self.log_message, default={}) \
                        .get("Dashboard", {}).get("5", {}).get("inds", {})
                    if not five_inds:
                        self.log_message(f"Skipping {sym}: no previous indicators found either.", False)
                        continue

                try:
                    current_bar_ts_5 = pd.to_datetime(five_inds.get("timestamp")).tz_convert(IST)
                except Exception:
                    self.log_message(f"Bad 5m timestamp for {sym}, skipping.", False)
                    continue

                new_candle_5 = (current_bar_ts_5 != self.last_bar_ts.get(sym))
                if not new_candle_5:
                    self.log_message(f"Waiting for new 5m candle for {sym}", True)
                    continue

                self.last_bar_ts[sym] = current_bar_ts_5
                self.log_message(f"New 5m candle for {sym} at {current_bar_ts_5.isoformat()}", False)

                # Get LTP (prefer WebSocket)
                ltp = self.get_websocket_ltp(sym) or five_inds.get("close")
                if not ltp:
                    self.log_message(f"Could not get LTP for {sym}, skipping strategies", False)
                    continue

                om = self.order_managers[sym]

                # ─────────────────────────── STRATEGY SELECTION ───────────────────────────
                om.execute_unified_strategy(
                    ltp=ltp,
                    all_inds=fresh_indicators_all_tfs.get(sym, {}),
                    primary_tf=selected_tf
                )

                # Log AI state for monitoring
                ai_state = om.get_ai_state()
                if ai_state.get("last_ai_action"):
                    self.log_message(f"[AI-CPR] {sym} State: {ai_state}", True)

            self.log_message("Cycle complete, sleeping for 2 seconds...", True)
            time.sleep(2)

    # ENHANCED: Add method to update AI strategy configuration dynamically
    def update_ai_strategy_config(self, symbol=None, enabled=None, min_conf=None, gate_trades=None):
        """
        Update AI CPR strategy configuration dynamically
        """
        symbols_to_update = [symbol] if symbol else self.symbols

        for sym in symbols_to_update:
            if sym in self.order_managers:
                self.order_managers[sym].update_ai_config(
                    enabled=enabled,
                    min_conf=min_conf,
                    gate_trades=gate_trades
                )
                self.log_message(f"[AI-CPR] Updated config for {sym}: enabled={enabled}, min_conf={min_conf}, gate_trades={gate_trades}", False)

    # ENHANCED: Add method to get AI status for monitoring
    def get_ai_status(self, symbol=None):
        """
        Get AI CPR status for monitoring/dashboard
        """
        status = {}
        symbols_to_check = [symbol] if symbol else self.symbols

        for sym in symbols_to_check:
            if sym in self.order_managers:
                status[sym] = {
                    "ai_state": self.order_managers[sym].get_ai_state(),
                    "position": self.order_managers[sym].position.get("type", "FLAT"),
                    "ai_enabled": self.order_managers[sym].AI_CPR_ENABLED
                }
        return status

    # ENHANCED: Add method to switch between strategy modes
    def set_strategy_mode(self, trend_strategy=None, ai_cpr_strategy=None, combined_strategy=None):
        """
        Switch between different strategy modes
        """
        if trend_strategy is not None:
            self.USE_TREND_STRATEGY = trend_strategy
            self.log_message(f"[STRATEGY] Trend strategy {'enabled' if trend_strategy else 'disabled'}", False)

        if ai_cpr_strategy is not None:
            self.USE_AI_CPR_STRATEGY = ai_cpr_strategy
            # Update all order managers
            for sym, om in self.order_managers.items():
                om.AI_CPR_ENABLED = ai_cpr_strategy
            self.log_message(f"[STRATEGY] AI CPR strategy {'enabled' if ai_cpr_strategy else 'disabled'}", False)

        if combined_strategy is not None:
            self.USE_COMBINED_STRATEGY = combined_strategy
            self.log_message(f"[STRATEGY] Combined strategy {'enabled' if combined_strategy else 'disabled'}", False)

    # ENHANCED: Override the process_pivots method to include AI analysis
    def process_pivots(self):
        """Enhanced pivot processing with AI context"""
        df = self.fetch_ohlc(self.symbol, "D", 30)
        if df.empty:
            return {}
        raw = self.indicator_calculator.calculate_pivot_points(df)
        clean = convert_dict_to_serializable(raw)

        # Add AI context if available
        if hasattr(self, 'ai_predictor') and self.ai_predictor:
            try:
                # Get current indicators for AI context
                current_inds = self.indicator_calculator.calculate_indicators(self.symbol, "5", pivot_data=raw)
                if "error" not in current_inds:
                    ai_analysis = analyze_cpr_strategy(current_inds, raw, self.ai_predictor)
                    clean["ai_context"] = {
                        "timestamp": dt.datetime.now(IST).isoformat(),
                        "analysis": ai_analysis
                    }
            except Exception as e:
                self.log_message(f"[AI-CPR] Pivot context error: {e}", True)

        data = { self.symbol: {"ts": dt.datetime.now(IST).isoformat(), **clean} }
        robust_save_json(data, self.pivot_json, self.log_message)
        return data

    # ENHANCED: Add method for AI model health check
    def check_ai_model_health(self):
        """
        Check if AI model is loaded and healthy
        """
        if not hasattr(self, 'ai_predictor') or not self.ai_predictor:
            return {"status": "error", "message": "AI predictor not initialized"}

        try:
            if self.ai_predictor.model is None:
                return {"status": "error", "message": "AI model not loaded"}

            return {
            "status": "healthy",
            "model_loaded": True,
            "message": "AI model loaded successfully - test prediction skipped for live trading"
            # "test_prediction": prediction,  # Commented for live
            # "test_confidence": confidence,  # Commented for live
            # "feature_shape": features.shape if features is not None else None  # Commented for live
        }

        except Exception as e:
            return {"status": "error", "message": f"AI model test failed: {str(e)}"}

    # ENHANCED: Modify the existing analyze_setup_score to include AI analysis
    def analyze_setup_score(self, tf):
        """Enhanced setup analysis with AI CPR integration"""
        inds = self.indicator_calculator.calculate_indicators(self.symbol, tf)
        if "error" in inds:
            return {
                "type": "NO_DATA",
                "score": 0,
                "summary": inds.get("error"),
                "ts": dt.datetime.now(IST).isoformat()
            }

        score = 0
        reasons = []

        # Existing trend analysis
        if inds.get("st21Trend") == 1:
            score += 2
            reasons.append("ST21 Bull")

        # NEW: AI CPR analysis
        cpr_analysis = inds.get("cpr_analysis", {})
        if cpr_analysis and "error" not in cpr_analysis:
            ai_label = cpr_analysis.get("ai_cpr_label")
            ai_confidence = cpr_analysis.get("ai_confidence", 0)
            ai_filter_pass = cpr_analysis.get("ai_filter_pass", False)

            if ai_label and ai_confidence > 0.6:
                if ai_label.upper() == "BUY" and ai_filter_pass:
                    score += 3
                    reasons.append(f"AI CPR BUY (conf: {ai_confidence:.2f})")
                elif ai_label.upper() == "SELL" and ai_filter_pass:
                    score -= 3
                    reasons.append(f"AI CPR SELL (conf: {ai_confidence:.2f})")
                elif ai_label.upper() in ["HOLD", "NEUTRAL"]:
                    score += 0
                    reasons.append(f"AI CPR HOLD (conf: {ai_confidence:.2f})")

        reco = ("STRONG_BUY"  if score >  3.5 else
                "STRONG_SELL" if score < -3.5 else
                "NEUTRAL")

        return {
            "type":   reco,
            "score":  round(score, 2),
            "reason": ", ".join(reasons),
            "ts":     dt.datetime.now(IST).isoformat(),
            "ai_analysis": cpr_analysis  # Include full AI analysis
        }

    # ── FS paths ────────────────────────────────────────────────────────────────

    def _setup_paths(self):
        self.data_paths = {}
        self.log_paths  = {}
        for sym in self.symbols:
            sym_clean = sym.replace(':', '_')
            data_dir = os.path.join(os.getcwd(), "data_bot")
            log_dir  = os.path.join(os.getcwd(), "logs_bot")
            report_dir = os.path.join(os.getcwd(), "reports_bot", sym_clean)
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(report_dir, exist_ok=True)
            self.data_paths[sym] = {
                'raw_api_log':   os.path.join(data_dir, f"fyers_raw_{sym_clean}.json"),
                'om_state_path': os.path.join(data_dir, f"om_state_{sym_clean}.json"),
                'om_event_log':  os.path.join(data_dir, f"om_events_{sym_clean}.json"),
                'pivot_json':    os.path.join(data_dir, f"pivot_{sym_clean}.json"),
                'all_data_json': os.path.join(data_dir, f"all_data_{sym_clean}.json"),
                'trade_report_dir': report_dir
            }
            self.log_paths[sym] = {
                'output_log':    os.path.join(log_dir, f"bot_log_{sym_clean}.txt"),
                'status_change': os.path.join(log_dir, f"status_change_{sym_clean}.txt")
            }

        # Main symbol aliases
        main = self.symbol
        self.raw_api_log   = self.data_paths[main]['raw_api_log']
        self.om_state_path = self.data_paths[main]['om_state_path']
        self.om_event_log  = self.data_paths[main]['om_event_log']
        self.pivot_json    = self.data_paths[main]['pivot_json']
        self.all_data_json = self.data_paths[main]['all_data_json']
        self.output_log    = self.log_paths[main]['output_log']
        self.status_change = self.log_paths[main]['status_change']

    def _initialize_symbol_files(self):
        for sym in self.symbols:
            # Data files
            for k, path in self.data_paths[sym].items():
                if k == "trade_report_dir":
                    os.makedirs(path, exist_ok=True)
                    continue
                if not os.path.exists(path):
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump({}, f)
            # Log files
            for _, path in self.log_paths[sym].items():
                if not os.path.exists(path):
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("")

    def _load_config(self, config_dir):
        for p in [os.path.join(os.getcwd(), config_dir), os.getcwd(), config_dir]:
            cid = os.path.join(p, "client_id.txt")
            tok = os.path.join(p, "access_token.txt")
            if os.path.exists(cid) and os.path.exists(tok):
                with open(cid) as f:   self.client_id    = f.read().strip()
                with open(tok) as f:   self.access_token = f.read().strip()
                self.log_message(f"Loaded config from {p}", False)
                return
        raise FileNotFoundError("Missing client_id.txt or access_token.txt")

    # ── Websocket ───────────────────────────────────────────────────────────────

    def _setup_websocket(self):
        try:
            self.fyers_websocket = data_ws.FyersDataSocket(
                access_token=f"{self.client_id}:{self.access_token}",
                log_path="",
                litemode=False,
                write_to_file=False,
                reconnect=True,
                on_connect=self._on_websocket_open,
                on_close=self._on_websocket_close,
                on_error=self._on_websocket_error,
                on_message=self._on_websocket_message
            )
            self.websocket_thread = threading.Thread(target=self._start_websocket, daemon=True)
            self.websocket_thread.start()
            self.log_message("WebSocket initialized", False)
        except Exception as e:
            self.log_message(f"WebSocket setup failed: {e}", False)

    def _start_websocket(self):
        try:
            self.fyers_websocket.connect()
        except Exception as e:
            self.log_message(f"WebSocket connection failed: {e}", False)

    def _on_websocket_open(self):
        try:
            data_type = "SymbolUpdate"
            self.fyers_websocket.subscribe(symbols=self.symbols, data_type=data_type)
            self.fyers_websocket.keep_running()
            self.log_message(f"WebSocket subscribed to {self.symbols}", False)
        except Exception as e:
            self.log_message(f"WebSocket subscription failed: {e}", False)

    def add_websocket_symbol(self, symbol):
        try:
            self.fyers_websocket.subscribe(symbols=[symbol], data_type="SymbolUpdate")
            self.log_message(f"Added symbol to WebSocket: {symbol}", False)
        except Exception as e:
            self.log_message(f"Failed to add symbol {symbol}: {e}", False)

    def remove_websocket_symbol(self, symbol):
        try:
            self.fyers_websocket.unsubscribe(symbols=[symbol])
            self.log_message(f"Removed symbol from WebSocket: {symbol}", False)
        except Exception as e:
            self.log_message(f"Failed to remove symbol {symbol}: {e}", False)

    def _on_websocket_message(self, message):
        try:
            if isinstance(message, dict) and 'symbol' in message and 'ltp' in message:
                symbol = message['symbol']
                ltp = message['ltp']
                now = dt.datetime.now(self.IST)

                # Update LTP cache
                with self.websocket_lock:
                    self.websocket_ltp[symbol] = {
                        'ltp': ltp,
                        'timestamp': now.isoformat()
                    }

                # Process OHLC accumulation
                self._process_ohlc_data(message)

                if self.DEBUG:
                    last_print = self._last_print_time.get(symbol)
                    if last_print is None or (now - last_print).total_seconds() > 5:
                        print(f"WebSocket LTP: {symbol} = {ltp}")
                        self._last_print_time[symbol] = now
        except Exception as e:
            self.log_message(f"WebSocket message error: {e}", False)


    def _process_ohlc_data(self, message):
        try:
            ms = message.get('exch_feed_time')
            if ms is None:
                return

            # Handle ms or sec epoch
            ts_sec = ms / 1000.0 if ms > 1e11 else float(ms)
            curr_time = dt.datetime.fromtimestamp(ts_sec, tz=self.IST)

            # Increment once per new minute
            minute_key = curr_time.replace(second=0, microsecond=0)
            if self._last_minute_seen != minute_key:
                self._last_minute_seen = minute_key
                self.timeframe_counter += 1

            # Close bar on timeframe boundary
            if self.timeframe_counter >= int(self.timeframe):
                with self.ohlc_lock:
                    for symbol in list(self.ohlc_data.keys()):
                        try:
                            if not self.ohlc_data[symbol]:
                                continue
                            high = max(self.ohlc_data[symbol])
                            low  = min(self.ohlc_data[symbol])
                            open_price  = self.ohlc_data[symbol][0]
                            close_price = self.ohlc_data[symbol][-1]

                            csv_dict = {
                                'minute': curr_time.strftime("%Y-%m-%d %H:%M:00"),
                                'symbol': str(symbol),
                                'open': float(open_price),
                                'high': float(high),
                                'low':  float(low),
                                'close':float(close_price)
                            }

                            self.csv_data.setdefault(symbol, []).append(csv_dict)
                            self._save_ohlc_csv(symbol, csv_dict, timeframe=str(self.timeframe))
                            self.log_message(f"OHLC CSV updated for {symbol}: {csv_dict}", True)
                        except Exception as e:
                            self.log_message(f"OHLC processing error for {symbol}: {e}", False)

                    self.ohlc_data = {}
                self.timeframe_counter = 0
            else:
                # Append LTP for this symbol
                symbol = message['symbol']
                with self.ohlc_lock:
                    self.ohlc_data.setdefault(symbol, []).append(float(message['ltp']))

        except Exception as e:
            self.log_message(f"OHLC processing error: {e}", False)

    def _save_ohlc_csv(self, symbol, csv_dict, timeframe="15"):
        try:
            csv_filename = f'{symbol.replace(":", "_")}_websocket_ohlc_alltest_{timeframe}min.csv'
            csv_path = os.path.join(os.getcwd(), "data_bot", csv_filename)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            file_exists = os.path.isfile(csv_path)
            csv_dict['timeframe'] = timeframe
            with open(csv_path, 'a', newline='') as f:
                field_names = ['minute', 'symbol', 'open', 'high', 'low', 'close', 'timeframe']
                writer = DictWriter(f, fieldnames=field_names)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_dict)
        except Exception as e:
            self.log_message(f"CSV save error for {symbol}: {e}", False)

    def _on_websocket_error(self, message):
        self.log_message(f"WebSocket error: {message}", False)

    def _on_websocket_close(self, message):
        self.log_message(f"WebSocket closed: {message}", False)

    def get_websocket_ltp(self, symbol, timeout=5):
        """Get LTP from WebSocket cache only (no REST fallback)."""
        try:
            with self.websocket_lock:
                if symbol in self.websocket_ltp:
                    data = self.websocket_ltp[symbol]
                    if data['timestamp']:
                        ts = dt.datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                        if (dt.datetime.now(IST) - ts).total_seconds() <= timeout:
                            return data['ltp']
            return None
        except Exception as e:
            self.log_message(f"WebSocket LTP error for {symbol}: {e}", False)
            return None

    # ── Logging helpers ─────────────────────────────────────────────────────────

    def log_message(self, msg, debug_only=False):
        """
        🔥 ENHANCED: Add gap/options context
        """
        if debug_only and not self.DEBUG:
            return

        ts = dt.datetime.now(self.IST).strftime("%Y-%m-%d %H:%M:%S %Z")

        # Add context tags
        tags = []
        if "[GAP" in msg.upper():
            tags.append("🚧")
        if "[OBX" in msg.upper() or "OPTION" in msg.upper():
            tags.append("📊")

        prefix = " ".join(tags) if tags else ""
        entry = f"{prefix} {ts} - {msg}"
        print(entry)

        try:
            with open(self.output_log, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except:
            pass



    def log_status_change(self, tf, prev, curr, inds):
        ts    = dt.datetime.now(self.IST).strftime("%Y-%m-%d %H:%M:%S %Z")
        lines = [f"[{ts}] TF={tf}"]
        for k in curr:
            if prev.get(k) != curr[k]:
                lines.append(f"  {k}: {prev.get(k)} -> {curr[k]}")
        if len(lines) > 1:
            try:
                with open(self.status_change, "a") as f:
                    f.write("\\n".join(lines) + "\\n\\n")
                self.log_message(f"Status change {tf}", True)
            except:
                pass

    # ── History fetch (with future-candle filter) ──────────────────────────────

    def fetch_ohlc(self, symbol, tf, days):
        try:
            to = dt.date.today()
            frm = to - dt.timedelta(days=int(days))
            resp = call_with_rate_limit_retry(
                self.fyers_sdk_instance.history,
                data={
                    "symbol": symbol,
                    "resolution": str(tf),
                    "date_format": "1",
                    "range_from": frm.strftime("%Y-%m-%d"),
                    "range_to":   to.strftime("%Y-%m-%d"),
                    "cont_flag": "1"
                }
            )
            if resp and resp.get("s") == "ok" and resp.get("candles"):
                df = pd.DataFrame(resp["candles"], columns=["Ts", "Open", "High", "Low", "Close", "Volume"])
                df["Timestamp"] = pd.to_datetime(df["Ts"], unit="s", utc=True).dt.tz_convert(IST)
                df.set_index("Timestamp", inplace=True)

                now_ts = pd.Timestamp.now(tz=IST)
                original_rows = len(df)
                df = df[df.index <= now_ts]
                filtered_rows = len(df)
                if original_rows > filtered_rows:
                    self.log_message(f"[DATA FIX] Removed {original_rows - filtered_rows} future-dated candles for {symbol}.", False)

                if df.empty:
                    self.log_message(f"[WARN] No historical data left for {symbol} after filtering future dates.", False)
                    return pd.DataFrame()

                for c in ["Open", "High", "Low", "Close", "Volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                return df.sort_index()

            self.log_message(f"OHLC error {symbol} {tf}: {resp.get('message','') if resp else 'No response'}", False)
        except Exception as e:
            self.log_message(f"OHLC exception {symbol} {tf}: {e}", False)
        return pd.DataFrame()

    # ── Pivots & ADX snapshots (optional dashboard) ────────────────────────────

    def process_pivots(self):
        df = self.fetch_ohlc(self.symbol, "D", 30)
        if df.empty:
            return {}
        raw   = self.indicator_calculator.calculate_pivot_points(df)
        clean = convert_dict_to_serializable(raw)
        data  = { self.symbol: {"ts": dt.datetime.now(IST).isoformat(), **clean} }
        robust_save_json(data, self.pivot_json, self.log_message)
        return data

    def initialize_pivots(self):
        piv_data = robust_load_json(self.pivot_json, self.log_message, default=None, debug_only=False)
        if not isinstance(piv_data, dict) or self.symbol not in piv_data:
            self.log_message("Generating new pivots…", False)
            self.process_pivots()
        else:
            self.log_message("Using existing pivots", True)

    def fetch_and_store_adx(self):
        df = fetchOHLC1(self.symbol, interval="5", duration=3)
        if df is not None and not df.empty:
            bundle = adx_efi_mom_trade_signal(df, self.symbol)
            payload = {
                "ts":  dt.datetime.now(IST).isoformat(),
                "sig": bundle[0],
                "ADX": bundle[1],
                "DI+": bundle[2],
                "DI-": bundle[3],
                "Mom": bundle[4],
                "EFI": bundle[5],
                "RSI": bundle[6]
            }
            all_data = robust_load_json(self.all_data_json, self.log_message, default={})
            all_data["ADX"] = convert_dict_to_serializable(payload)
            robust_save_json(all_data, self.all_data_json, self.log_message)

    def analyze_setup_score(self, tf):
        inds = self.indicator_calculator.calculate_indicators(self.symbol, tf)
        if "error" in inds:
            return {
                "type": "NO_DATA",
                "score": 0,
                "summary": inds.get("error"),
                "ts": dt.datetime.now(IST).isoformat()
            }
        score   = 0
        reasons = []
        if inds.get("st21Trend") == 1:
            score += 2
            reasons.append("ST21 Bull")
        reco = ("STRONG_BUY"  if score >  3.5 else
                "STRONG_SELL" if score < -3.5 else
                "NEUTRAL")
        return {
            "type":   reco,
            "score":  round(score, 2),
            "reason": ", ".join(reasons),
            "ts":     dt.datetime.now(IST).isoformat()
        }

    def _ai_place_order(self, side, sym, indicators):
        print("DEBUG: Entered _ai_place_order")
        om = self.order_managers[sym]
        trade_qty = 1

        ema5 = indicators.get("ema_5")
        ema21 = indicators.get("ema_21")
        ltp = indicators.get("close")
        trend = indicators.get("trend")  # ← optional: higher timeframe trend flag
        last_pos = om.position.get("last_type", None)
        current_pos = om.position.get("type", "FLAT")

        # ✅ Guard: If indicators are missing, skip
        if ema5 is None or ema21 is None or ltp is None:
            self.log_message(
                f"[AI-CPR] Skipping order for {sym}: indicators missing (ema5={ema5}, ema21={ema21}, ltp={ltp})", True
            )
            return "SKIP - Indicators missing"

        # ✅ Normalize side
        if side in ["Bullish", "BUY"]:
            side = "BUY"
        elif side in ["Bearish", "SELL"]:
            side = "SELL"
        else:
            if current_pos != "FLAT":
                return "HOLD"
            return "FLAT - No trade"

        # ✅ Optional Trend Filter (skip entries against trend)
        if trend:
            if side == "BUY" and trend != "UP":
                self.log_message(f"[AI-CPR] Skipping BUY for {sym}: Trend is {trend}", True)
                return "SKIP - Against trend"
            if side == "SELL" and trend != "DOWN":
                self.log_message(f"[AI-CPR] Skipping SELL for {sym}: Trend is {trend}", True)
                return "SKIP - Against trend"

        # ✅ New Entry
        if current_pos == "FLAT":
            if side == "BUY" and ema5 > ema21 and last_pos != "BUY":
                om.ai_buy(sym, trade_qty)
                return "BUY"
            elif side == "SELL" and ema5 < ema21 and last_pos != "SELL":
                om.ai_sell(sym, trade_qty)
                return "SELL"
            return "FLAT - No trade"

        # ✅ Exit on direction change
        if (current_pos == "BUY" and side == "SELL") or (current_pos == "SELL" and side == "BUY"):
            self.log_message(f"[AI-CPR] EXIT: Direction change for {sym}", True)
            om.ai_exit_all(sym)
            return "EXIT on direction change"

        # ✅ Hold if already in same side
        if (current_pos == side) or (current_pos == "FLAT" and last_pos == side):
            self.log_message(f"[AI-CPR] HOLD: Already in {side} or just exited {side} for {sym}", True)
            return "HOLD"

        return "HOLD"

    def _check_recent_trend(self, df, i, recent_candles):
        uptrend, downtrend = True, True
        for k in range(recent_candles):
            idx = i - k
            if not (df['Close'][idx] > df['Open'][idx]):
                uptrend = False
            if not (df['Close'][idx] < df['Open'][idx]):
                downtrend = False
            if k < recent_candles - 1:
                prev_idx = i - (k + 1)
                if not (df['High'][idx] > df['High'][prev_idx] and df['Low'][idx] > df['Low'][prev_idx]):
                    uptrend = False
                if not (df['High'][idx] < df['High'][prev_idx] and df['Low'][idx] < df['Low'][prev_idx]):
                    downtrend = False
        return uptrend, downtrend

    def _check_preceding_trend(self, df, i, recent_candles, preceding_candles):
        uptrend, downtrend = True, True
        start_idx = i - recent_candles
        for k in range(preceding_candles):
            idx = start_idx - k
            if not (df['Close'][idx] > df['Open'][idx]):
                downtrend = False
            if not (df['Close'][idx] < df['Open'][idx]):
                uptrend = False
            if k < preceding_candles - 1:
                prev_idx = start_idx - (k + 1)
                if not (df['High'][idx] < df['High'][prev_idx] and df['Low'][idx] < df['Low'][prev_idx]):
                    uptrend = False
                if not (df['High'][idx] > df['High'][prev_idx] and df['Low'][idx] > df['Low'][prev_idx]):
                    downtrend = False
        return uptrend, downtrend


    # (Optional) monitoring helpers
    def _log_ohlc_status(self):
        try:
            with self.ohlc_lock:
                status = {
                    'timeframe': self.timeframe,
                    'counter': self.timeframe_counter,
                    'active_symbols': list(self.ohlc_data.keys()),
                    'symbol_counts': {symbol: len(data) for symbol, data in self.ohlc_data.items()},
                    'csv_records': {symbol: len(data) for symbol, data in self.csv_data.items()}
                }
            self.log_message(f"OHLC Status: {status}", True)
        except Exception as e:
            self.log_message(f"OHLC status error: {e}", False)

    def get_websocket_status(self):
        try:
            with self.websocket_lock:
                ltp_status = {
                    symbol: {
                        'ltp': data['ltp'],
                        'age_seconds': (dt.datetime.now(IST) - dt.datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))).total_seconds()
                    }
                    for symbol, data in self.websocket_ltp.items()
                }

            with self.ohlc_lock:
                ohlc_status = {
                    'timeframe': self.timeframe,
                    'counter': self.timeframe_counter,
                    'active_symbols': list(self.ohlc_data.keys()),
                    'symbol_counts': {symbol: len(data) for symbol, data in self.ohlc_data.items()},
                    'csv_records': {symbol: len(data) for symbol, data in self.csv_data.items()}
                }

            return {
                'ltp_data': ltp_status,
                'ohlc_data': ohlc_status,
                'websocket_active': hasattr(self, 'fyers_websocket') and self.fyers_websocket is not None
            }
        except Exception as e:
            self.log_message(f"WebSocket status error: {e}", False)
            return {'error': str(e)}

# ───────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ───────────────────────────────────────────────────────────────────────────────

def main():
    try:
        bot = TradingBot()
        # Check gap state
        for sym in bot.symbols:
            om = bot.order_managers[sym]
            gap_state = om.gap.state
            print(f"Gap state for {sym}: {gap_state}")
        # Trade based on 1-min selected TF for decision cycle; 5m/15m used inside OM
        bot.run(selected_tf=tf_selected)
    except Exception as e:
        ts = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{ts}] FATAL: {e}\\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
