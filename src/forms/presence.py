"""
presence.py: days of physical presence in the United States, for Form 8843 line 4a.

Counting rule (IRS Pub 519, "Substantial Presence Test"): you are treated as present in the
United States on any day you are physically present there at any time during the day. So the
day you fly out and the day you land both count as days present; only the days strictly between
a departure and the matching return are days abroad.

Inputs come from the student, never from a model:
    first_entry   the first day the student was in the U.S. in this stay (Form 8843 line 1a date)
    trips         trips abroad after first_entry: (left_us, returned_to_us) date pairs
    year          the calendar year to count
    until         last day to count (default: Dec 31 of year); a student who left the U.S. for
                  good passes the departure date here

The result is shown to the student as "computed, please confirm" (Form 8843 line 4a): a wrong
number here changes the substantial presence test, so the student checks it against their
I-94 travel history before the final PDF is made.
"""
from __future__ import annotations

from datetime import date, timedelta


def count_days_present(first_entry: date, trips: list[tuple[date, date]], year: int,
                       until: date | None = None) -> int:
    """Number of days in `year` on which the student was in the U.S. at some point.

    Example: first_entry 2024-08-15, one trip (2024-12-20, 2025-01-10), year 2025
        Jan 1-9 2025 are abroad (strictly between Dec 20 and Jan 10); Jan 10 counts (landed).
        -> 365 - 9 = 356

    Cases to handle:
        - first_entry after `year`                    -> 0
        - first_entry inside `year`                   -> count from first_entry, not Jan 1
        - a trip that starts in one year and ends in the next (only the part inside `year` counts)
        - departure day and return day are days present
        - `until` earlier than Dec 31 (left the U.S. for good)

    Trips are already validated by the caller: returned_to_us > left_us, trips do not overlap,
    and every trip starts on or after first_entry.
    """
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
