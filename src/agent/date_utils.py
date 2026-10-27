"""Date calculation utilities for ZentroMart VoiceOps.

Rules:
- Reference "Today" date: 2026-10-20.
- Business days: Monday through Saturday (Sunday is weekend off).
- Holidays: dates in holidays.csv are skipped.
- Return windows: calendar days.
"""
from datetime import date, datetime, timedelta
from typing import List, Set
import csv
import os

REFERENCE_DATE = date(2026, 10, 20)

def load_holidays(data_dir: str = None) -> Set[date]:
    """Load warehouse holidays from data/holidays.csv."""
    if data_dir is None:
        data_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Capstone_Project_Dataset_FTTT_2026", "data")
        )
    holidays_file = os.path.join(data_dir, "holidays.csv")
    holidays = set()
    if os.path.exists(holidays_file):
        with open(holidays_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d_str = row.get("date", "").strip()
                if d_str:
                    holidays.add(datetime.strptime(d_str, "%Y-%m-%d").date())
    return holidays

HOLIDAYS = load_holidays()

def is_business_day(d: date, holidays: Set[date] = None) -> bool:
    """Monday=0 to Saturday=5 are business days, Sunday=6 is not. Also exclude holidays."""
    if holidays is None:
        holidays = HOLIDAYS
    if d.weekday() == 6:  # Sunday
        return False
    if d in holidays:
        return False
    return True

def add_business_days(start_date: date, num_days: int, holidays: Set[date] = None) -> date:
    """Add N business days starting from start_date."""
    if holidays is None:
        holidays = HOLIDAYS
    cur = start_date
    added = 0
    while added < num_days:
        cur += timedelta(days=1)
        if is_business_day(cur, holidays):
            added += 1
    return cur

def count_business_days(start_date: date, end_date: date, holidays: Set[date] = None) -> int:
    """Count business days between start_date (exclusive) and end_date (inclusive)."""
    if holidays is None:
        holidays = HOLIDAYS
    if start_date >= end_date:
        return 0
    cur = start_date + timedelta(days=1)
    count = 0
    while cur <= end_date:
        if is_business_day(cur, holidays):
            count += 1
        cur += timedelta(days=1)
    return count

def add_calendar_days(start_date: date, num_days: int) -> date:
    """Add N calendar days (used for return windows)."""
    return start_date + timedelta(days=num_days)

def parse_iso_date(date_str: str) -> date:
    """Parse YYYY-MM-DD or return None."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except Exception:
        return None
