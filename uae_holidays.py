# pip install hijri-converter python-dateutil
from datetime import date, timedelta
from dateutil.rrule import rrule, DAILY
from hijri_converter import convert
import logging
from typing import Set, Dict, List, Optional

logger = logging.getLogger(__name__)

def _date_window(center: date, days: int) -> Set[date]:
    """Inclusive ±days window around center -> set[date]."""
    start, end = center - timedelta(days=days), center + timedelta(days=days)
    return {dt.date() for dt in rrule(DAILY, dtstart=start, until=end)}

def _gregorian_anchors_for_islamic_year(hijri_year: int) -> List[date]:
    """
    Return Gregorian anchor dates (as date objects) for UAE Islamic holidays
    for a given Hijri year. Eids include all days in their typical spans.
    """
    anchors = []

    def add(hy, hm, hd, span_days=1):
        g = convert.Hijri(hy, hm, hd).to_gregorian()
        g0 = date(g.year, g.month, g.day)
        for i in range(span_days):
            anchors.append(g0 + timedelta(days=i))

    # Hijri New Year (1 Muharram) – 1 day
    add(hijri_year, 1, 1, span_days=1)

    # Prophet Muhammad's Birthday (12 Rabi' al-Awwal) – 1 day
    add(hijri_year, 3, 12, span_days=1)

    # Eid al-Fitr (1–3 Shawwal) – commonly 3 days
    add(hijri_year, 10, 1, span_days=3)

    # Arafat Day (9 Dhu al-Hijjah) – 1 day
    add(hijri_year, 12, 9, span_days=1)

    # Eid al-Adha (10–12 Dhu al-Hijjah) – commonly 3 days
    add(hijri_year, 12, 10, span_days=3)

    return anchors

def _uae_fixed_anchors_for_year(g_year: int) -> List[date]:
    """Fixed-date UAE holidays in Gregorian calendar."""
    return [
        date(g_year, 1, 1),   # New Year's Day
        date(g_year, 12, 1),  # Commemoration Day
        date(g_year, 12, 2),  # National Day (day 1)
        date(g_year, 12, 3),  # National Day (day 2)
    ]

def uae_holiday_windows(
    year: int,
    pad_days: int = 7,
    include_fixed: bool = True,
    include_islamic: bool = True,
) -> Dict[str, Set[date]]:
    """
    Build a dict[str, set[date]] of ±pad_days windows for UAE holidays whose
    *windows* intersect the given Gregorian year. Handles Islamic year shifts.

    Keys:
      - "New Year's Day", "Commemoration Day", "National Day (1/2)"
      - "Hijri New Year", "Mawlid", "Eid al-Fitr", "Arafat Day", "Eid al-Adha"
    """
    named_windows = {}

    # --- Collect anchors from (year-1, year, year+1) so spillovers are caught ---
    anchors_named = []

    if include_fixed:
        for y in (year - 1, year, year + 1):
            fixed = _uae_fixed_anchors_for_year(y)
            anchors_named.extend([
                ("New Year's Day", fixed[0]),
                ("Commemoration Day", fixed[1]),
                ("National Day (1)", fixed[2]),
                ("National Day (2)", fixed[3]),
            ])

    if include_islamic:
        # Figure out which Hijri years overlap our 3-year window
        probe_days = [date(y, m, 15) for y in (year - 1, year, year + 1) for m in (2, 6, 10, 12)]
        hijri_years = {convert.Gregorian(d.year, d.month, d.day).to_hijri().year for d in probe_days}

        for hy in sorted(hijri_years):
            for g in _gregorian_anchors_for_islamic_year(hy):
                # Map each Gregorian anchor back to its Islamic holiday name
                # by reverse-converting and checking HM/HD
                h = convert.Gregorian(g.year, g.month, g.day).to_hijri()
                name = None
                if (h.month, h.day) == (1, 1):
                    name = "Hijri New Year"
                elif (h.month, h.day) == (3, 12):
                    name = "Mawlid"
                elif (h.month, h.day) in [(10, 1), (10, 2), (10, 3)]:
                    name = "Eid al-Fitr"
                elif (h.month, h.day) == (12, 9):
                    name = "Arafat Day"
                elif (h.month, h.day) in [(12, 10), (12, 11), (12, 12)]:
                    name = "Eid al-Adha"

                if name:
                    anchors_named.append((name, g))

    # --- Build windows and keep only dates inside the target Gregorian year ---
    start_y, end_y = date(year, 1, 1), date(year, 12, 31)
    for name, anchor in anchors_named:
        win = _date_window(anchor, pad_days)
        # Filter to the target year
        win_in_year = {d for d in win if start_y <= d <= end_y}
        if not win_in_year:
            continue
        named_windows.setdefault(name, set()).update(win_in_year)

    return named_windows

def is_uae_holiday(check_date: date, pad_days: int = 7) -> bool:
    """
    Check if a date falls within ±pad_days of any UAE holiday.
    
    Args:
        check_date: The date to check
        pad_days: Number of days before/after holiday to consider as holiday period
        
    Returns:
        True if date is within holiday window, False otherwise
    """
    try:
        windows = uae_holiday_windows(check_date.year, pad_days=pad_days)
        all_holiday_dates = set().union(*windows.values()) if windows else set()
        return check_date in all_holiday_dates
    except Exception as e:
        logger.error(f"Error checking UAE holiday: {e}")
        return False

def get_holiday_names(check_date: date, pad_days: int = 7) -> List[str]:
    """
    Get list of holiday names if date falls within their windows.
    
    Args:
        check_date: The date to check
        pad_days: Number of days before/after holiday to consider as holiday period
        
    Returns:
        List of holiday names that this date falls within
    """
    try:
        windows = uae_holiday_windows(check_date.year, pad_days=pad_days)
        return [name for name, dates in windows.items() if check_date in dates]
    except Exception as e:
        logger.error(f"Error getting holiday names: {e}")
        return []

def is_working_day(check_date: date, holiday_pad_days: int = 7) -> bool:
    """
    Check if a date is a working day in UAE (not weekend or holiday).
    
    Args:
        check_date: The date to check
        holiday_pad_days: Days before/after holiday to consider as non-working
        
    Returns:
        True if it's a working day, False if weekend or holiday
    """
    # Check if weekend (Friday = 4, Saturday = 5, Sunday = 6)
    if check_date.weekday() in [4, 5, 6]:
        return False
    
    # Check if holiday
    if is_uae_holiday(check_date, pad_days=holiday_pad_days):
        return False
    
    return True

def add_working_days(start_date: date, days: int, holiday_pad_days: int = 7) -> date:
    """
    Add working days to a date, skipping weekends and holidays.
    
    Args:
        start_date: Starting date
        days: Number of working days to add
        holiday_pad_days: Days before/after holiday to skip
        
    Returns:
        The resulting date after adding working days
    """
    current_date = start_date
    days_added = 0
    
    while days_added < days:
        current_date += timedelta(days=1)
        if is_working_day(current_date, holiday_pad_days):
            days_added += 1
    
    return current_date

def count_working_days(start_date: date, end_date: date, holiday_pad_days: int = 7) -> int:
    """
    Count working days between two dates (inclusive).
    
    Args:
        start_date: Starting date
        end_date: Ending date
        holiday_pad_days: Days before/after holiday to exclude
        
    Returns:
        Number of working days between dates
    """
    if start_date > end_date:
        return 0
    
    working_days = 0
    current_date = start_date
    
    while current_date <= end_date:
        if is_working_day(current_date, holiday_pad_days):
            working_days += 1
        current_date += timedelta(days=1)
    
    return working_days

def get_next_working_day(check_date: date, holiday_pad_days: int = 7) -> date:
    """
    Get the next working day from a given date.
    
    Args:
        check_date: The date to start from
        holiday_pad_days: Days before/after holiday to skip
        
    Returns:
        The next working day
    """
    next_date = check_date
    while not is_working_day(next_date, holiday_pad_days):
        next_date += timedelta(days=1)
    return next_date

def get_previous_working_day(check_date: date, holiday_pad_days: int = 7) -> date:
    """
    Get the previous working day from a given date.
    
    Args:
        check_date: The date to start from
        holiday_pad_days: Days before/after holiday to skip
        
    Returns:
        The previous working day
    """
    prev_date = check_date
    while not is_working_day(prev_date, holiday_pad_days):
        prev_date -= timedelta(days=1)
    return prev_date

# ---------- Example ----------
if __name__ == "__main__":
    YEAR = 2025
    windows = uae_holiday_windows(YEAR, pad_days=7)

    # Flatten if you just need a quick membership test:
    all_dates = set().union(*windows.values()) if windows else set()

    print(f"{YEAR} dates within ±7 days of ANY UAE holiday:", len(all_dates))
    for name, days in sorted(windows.items(), key=lambda x: min(x[1]) if x[1] else date(YEAR,1,1)):
        print(name, "->", f"{len(days)} days in window (example min/max: {min(days)} .. {max(days)})")

    # Usage example:
    some_date = date(2025, 6, 16)  # pick a date to test
    hit = [n for n, ds in windows.items() if some_date in ds]
    print(some_date, "is within a holiday window:", bool(hit), "; matches:", hit)
    
    # Test working days functions
    print("\nWorking days examples:")
    test_date = date(2025, 1, 1)
    print(f"{test_date} is working day:", is_working_day(test_date))
    print(f"Next working day after {test_date}:", get_next_working_day(test_date))
    
    # Add 5 working days
    result = add_working_days(test_date, 5)
    print(f"5 working days after {test_date}: {result}")