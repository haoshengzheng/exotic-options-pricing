"""
Trading-time and calendar-time utilities for the Chinese onshore market.

This module powers the dual-time framework used throughout the pricing library: option diffusion is driven by trading time (active exchange
sessions only), while discounting is driven by calendar time (continuous 365-day basis). Mixing these two consistently requires session-aware
helpers, which is what this module provides.

SESSIONS:
The current session layout reflects the most common pattern across Chinese commodity and financial futures:
    Day sessions   : 09:00-10:15, 10:30-11:30, 13:30-15:00
    Night session  : 21:00-23:00 (closing same day, no cross-midnight)
Total active seconds per full trading day: 20,700.
Trading days per year (working calendar):  242.

Limitations:
Different products in China have different session boundaries. This module hard-codes a single session layout and does
not yet support per-product schedules.
For products outside the default schedule, the constants `_DAY_SESSIONS`, `_NIGHT_START`, `_NIGHT_END`, and
`SECONDS_PER_FULL_TRADE_DAY` need to be adjusted manually. A future extension would parameterize this by contract symbol.
"""
from datetime import datetime, date, timedelta, time
from typing import Optional, List
import chinese_calendar

_DAY_SESSIONS: list[tuple[time, time]] = [
    (time(9,0), time(10,15)),
    (time(10,30),time(11,30)),
    (time(13,30),time(15,0))
]

_NIGHT_START = time(21,0)
_NIGHT_END = time(23,0)
SECONDS_PER_FULL_TRADE_DAY = 20700
trading_days_per_year = 242

def is_cn_trading_day(d: date) -> bool:
   if d.weekday() >=5:
       return False
   return not chinese_calendar.is_holiday(d)

def _in_day_sessions(t:time) -> bool:
     return any (s <= t <= e for s, e in _DAY_SESSIONS)

def prev_trading_day(d:date) -> date:
     cur = d - timedelta(days=1)
     while not is_cn_trading_day(cur):
         cur -= timedelta(days=1)
     return cur

def next_trading_day(d:date) -> date:
    cur = d + timedelta(days=1)
    while not is_cn_trading_day(cur):
        cur += timedelta(days=1)
    return cur

def next_trading_day_open(dt:datetime) -> datetime:
    cur = dt.date()
    next_dt = next_trading_day(cur)
    return datetime.combine(next_dt, _NIGHT_START)

def get_all_sessions_of_trading_day(d:date) -> list[tuple[datetime, datetime]]:
    """
    Return all (start, end) datetime intervals belonging to trading day `d`.
    By Chinese-futures convention, a "trading day" runs from the previous trading day's night session through the current day's afternoon close.
    If the time interval between the current trading day and the previous trading day is more than three days, the night session
    will be excluded because there will be a holiday during this period, and there are no night sessions during holidays.
    """
    if not is_cn_trading_day(d):
        return []
    sessions = []
    prev_td = prev_trading_day(d)
    if (d - prev_td).days <= 3:
        night_start_dt = datetime.combine(prev_td, _NIGHT_START)
        if _NIGHT_END < _NIGHT_START:
            night_end_dt = datetime.combine(prev_td + timedelta(days=1), _NIGHT_END)
        else:
            night_end_dt = datetime.combine(prev_td, _NIGHT_END)
        sessions.append((night_start_dt, night_end_dt))

    for s_time, e_time in _DAY_SESSIONS:
        sessions.append((datetime.combine(d, s_time), datetime.combine(d, e_time)))
    return sessions

def count_trading_seconds_precise(start_dt: datetime, end_dt: datetime) -> float:
    """Count active trading seconds between two timestamps."""
    total_seconds = 0
    curr_d = start_dt.date() - timedelta(days=1)
    scan_end_d  = end_dt.date() + timedelta(days=3)
    while curr_d <=scan_end_d:
        if is_cn_trading_day(curr_d):
            for s_dt, e_dt in get_all_sessions_of_trading_day(curr_d):
                overlap_start = max(s_dt, start_dt)
                overlap_end   = min(e_dt, end_dt)
                if overlap_start < overlap_end:
                    total_seconds += (overlap_end - overlap_start).total_seconds()
        curr_d += timedelta(days=1)
    return total_seconds


def get_trading_day(dt: datetime) -> Optional[date]:
    d, t = dt.date(), dt.time()
    cross_midnight = _NIGHT_END < _NIGHT_START
    if cross_midnight and t <= _NIGHT_END:
        prev_d = d - timedelta(days=1)
        if is_cn_trading_day(d) and is_cn_trading_day(prev_d):
            return d
        if not is_cn_trading_day(d):
            return next_trading_day(d)
        return d
    if _in_day_sessions(t):
        return d if is_cn_trading_day(d) else next_trading_day(d)
    if t < _DAY_SESSIONS[0][0]:
        return d if is_cn_trading_day(d) else next_trading_day(d)
    if t < _DAY_SESSIONS[-1][1]:
        return d if is_cn_trading_day(d) else next_trading_day(d)
    if t >= _NIGHT_START:
        nd = next_trading_day(d)
        return nd
    return next_trading_day(d)


def count_trading_days(start_dt: datetime, end_dt: datetime) -> int:
    start_td = get_trading_day(start_dt)
    end_td = get_trading_day(end_dt)
    if start_td > end_td:
        return 0
    count = 0
    cur = start_td
    while cur <= end_td:
        if is_cn_trading_day(cur):
            count += 1
        cur += timedelta(days=1)
    return count


def parse_dt(s: str) -> datetime:
    for fmt in ('%Y.%m.%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                '%Y.%m.%d', '%Y-%m-%d', '%Y/%m/%d'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime string: {s!r}")

def generate_trading_day_obs(start_dt:str, end_dt:str, obs_time:time = time(15,0)) -> List[str]:
    start = parse_dt(start_dt)
    end = parse_dt(end_dt)
    result = []
    cur_date = start.date()
    end_date = end.date()
    while cur_date <= end_date:
        if is_cn_trading_day(cur_date):
            obs_dt = datetime.combine(cur_date, obs_time)
            if start <= obs_dt <= end:
              result.append(obs_dt.strftime('%Y-%m-%d %H:%M:%S'))
        cur_date += timedelta(days=1)
    return result
