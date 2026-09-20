"""Date helpers shared across the data clients.

All Indian districts sit in a single timezone, so "today" and "yesterday"
are well defined from a fixed UTC+05:30 offset. A fixed offset is used
rather than zoneinfo("Asia/Kolkata") on purpose: zoneinfo needs the
`tzdata` package on Windows, which is where this project is developed,
and India has no DST for the offset to get wrong.
"""
from __future__ import annotations

import datetime as dt

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def today_ist() -> dt.date:
    return dt.datetime.now(IST).date()


def yesterday_ist() -> dt.date:
    """The most recent fully-elapsed day. Daily aggregates for today are
    still partial, so any historical series should end here."""
    return today_ist() - dt.timedelta(days=1)
