"""
presence.py: days present in the U.S. for Form 8843 line 4a.

Rule (IRS Pub 519): any day you are in the U.S. at any time counts, so departure and return days
both count; only days strictly between leaving and returning are abroad. The student confirms the
result against their I-94 history, since it decides the substantial presence test.
"""
from __future__ import annotations

from datetime import date, timedelta


def count_days_present(first_entry: date, trips: list[tuple[date, date]], year: int,
                       until: date | None = None) -> int:
    """trips: (left_us, returned_to_us) pairs, already validated (ordered, no overlap, after first_entry).
    until: last day to count, for a student who left for good (default Dec 31)."""
    start = max(date(year, 1, 1), first_entry)
    end = min(date(year, 12, 31), until or date(year, 12, 31))
    if start > end:
        return 0
    days = (end - start).days + 1
    for left, back in trips:
        away_from = max(left + timedelta(days=1), start)
        away_to = min(back - timedelta(days=1), end)
        if away_from <= away_to:
            days -= (away_to - away_from).days + 1
    return days
