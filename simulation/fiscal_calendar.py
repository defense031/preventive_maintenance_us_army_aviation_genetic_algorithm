"""Fiscal calendar system for aviation simulation.

Ported from: aviation_hierarchical_sim_v2/core_v2/fiscal_calendar.R

Fiscal Year (FY) runs October 1 - September 30.
Example: FY2025 = Oct 1, 2024 through Sep 30, 2025

Quarters:
- Q1: Oct-Dec
- Q2: Jan-Mar
- Q3: Apr-Jun
- Q4: Jul-Sep
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional


@dataclass
class FiscalDay:
    """Single day in the fiscal calendar."""

    # Simulation tracking
    sim_day: int  # 1-indexed simulation day

    # Calendar date
    date: date
    cal_year: int
    cal_month: int
    cal_month_name: str
    cal_day: int

    # Day of week
    dow: str  # "Monday", "Tuesday", etc.
    dow_num: int  # 1=Sunday, 7=Saturday (matches R's wday())
    is_weekend: bool
    is_weekday: bool

    # Fiscal year tracking
    fiscal_year: int  # FY2025 means Oct 2024 - Sep 2025
    fiscal_quarter: int  # 1-4
    fiscal_month: int  # 1-12 within fiscal year

    # Progress tracking
    days_into_fy: int  # Days since start of fiscal year
    days_into_quarter: int  # Days since start of quarter
    fy_progress: float  # 0.0-1.0, progress through fiscal year
    quarter_progress: float  # 0.0-1.0, progress through quarter

    # Period end flags
    is_month_end: bool
    is_quarter_end: bool
    is_fy_end: bool

    # Week helpers
    is_friday: bool
    is_monday: bool
    days_until_weekend: int

    # Budget period markers
    quarter_week: int  # Week number within quarter
    is_quarter_start: bool  # First 5 days of quarter
    is_quarter_final_month: bool  # Last month of quarter (days 61-90)

    # Federal holidays
    is_federal_holiday: bool


class FiscalCalendar:
    """Fiscal calendar for simulation.

    Handles FY/quarter/month tracking and provides lookup methods.
    """

    def __init__(self, start_fy: int = 2025, sim_days: int = 365):
        """Initialize fiscal calendar.

        Args:
            start_fy: Starting fiscal year (e.g., 2025 = Oct 1, 2024 - Sep 30, 2025)
            sim_days: Number of simulation days
        """
        self.start_fy = start_fy
        self.sim_days = sim_days

        # FY2025 starts October 1, 2024
        self.fy_start_date = date(start_fy - 1, 10, 1)

        # Generate calendar
        self.calendar: List[FiscalDay] = self._build_calendar()

        # Metadata
        self.metadata = {
            "start_date": self.fy_start_date,
            "end_date": self.fy_start_date + timedelta(days=sim_days - 1),
            "fiscal_years_covered": list(set(d.fiscal_year for d in self.calendar)),
            "total_days": sim_days,
        }

    def _build_calendar(self) -> List[FiscalDay]:
        """Build complete fiscal calendar."""
        calendar = []

        for i in range(self.sim_days):
            current_date = self.fy_start_date + timedelta(days=i)
            fiscal_day = self._build_fiscal_day(i + 1, current_date)  # 1-indexed
            calendar.append(fiscal_day)

        # Mark federal holidays
        self._mark_federal_holidays(calendar)

        return calendar

    def _build_fiscal_day(self, sim_day: int, current_date: date) -> FiscalDay:
        """Build FiscalDay object for a single date."""
        # Date components
        cal_year = current_date.year
        cal_month = current_date.month
        cal_day = current_date.day

        # Day of week (convert Python's 0=Monday to R's 1=Sunday)
        python_dow = current_date.weekday()  # 0=Monday, 6=Sunday
        dow_num = (python_dow + 2) % 7  # Convert to 1=Sunday, 7=Saturday
        if dow_num == 0:
            dow_num = 7

        dow_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        dow = dow_names[dow_num - 1]

        is_weekend = dow_num in (1, 7)  # Sunday or Saturday
        is_weekday = not is_weekend

        # Fiscal year (Oct-Sep)
        fiscal_year = cal_year + 1 if cal_month >= 10 else cal_year

        # Fiscal quarter
        if cal_month in (10, 11, 12):
            fiscal_quarter = 1  # Q1: Oct-Dec
        elif cal_month in (1, 2, 3):
            fiscal_quarter = 2  # Q2: Jan-Mar
        elif cal_month in (4, 5, 6):
            fiscal_quarter = 3  # Q3: Apr-Jun
        else:  # 7, 8, 9
            fiscal_quarter = 4  # Q4: Jul-Sep

        # Fiscal month (1-12 within FY)
        fiscal_month = ((cal_month - 10) % 12) + 1

        # Days into fiscal year
        days_into_fy = (current_date - self.fy_start_date).days + 1

        # Days into quarter (calculate quarter start)
        if fiscal_quarter == 1:
            quarter_start = date(cal_year, 10, 1)
        elif fiscal_quarter == 2:
            quarter_start = date(cal_year, 1, 1)
        elif fiscal_quarter == 3:
            quarter_start = date(cal_year, 4, 1)
        else:  # Q4
            quarter_start = date(cal_year, 7, 1)

        days_into_quarter = (current_date - quarter_start).days + 1

        # Progress tracking
        fy_progress = (days_into_fy - 1) / 365  # -1 because day 1 = 0% progress
        quarter_progress = (days_into_quarter - 1) / 90  # Approximate

        # Period end flags
        # Month end: current date is last day of month
        next_day = current_date + timedelta(days=1)
        is_month_end = next_day.month != cal_month

        is_quarter_end = is_month_end and cal_month in (12, 3, 6, 9)
        is_fy_end = cal_month == 9 and is_month_end

        # Week helpers
        is_friday = dow_num == 6
        is_monday = dow_num == 2
        days_until_weekend = 0 if is_weekend else (7 - dow_num)

        # Budget period markers
        quarter_week = (days_into_quarter + 6) // 7  # Ceiling division
        is_quarter_start = days_into_quarter <= 5
        is_quarter_final_month = days_into_quarter > 60

        return FiscalDay(
            sim_day=sim_day,
            date=current_date,
            cal_year=cal_year,
            cal_month=cal_month,
            cal_month_name=current_date.strftime("%B"),
            cal_day=cal_day,
            dow=dow,
            dow_num=dow_num,
            is_weekend=is_weekend,
            is_weekday=is_weekday,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            fiscal_month=fiscal_month,
            days_into_fy=days_into_fy,
            days_into_quarter=days_into_quarter,
            fy_progress=fy_progress,
            quarter_progress=quarter_progress,
            is_month_end=is_month_end,
            is_quarter_end=is_quarter_end,
            is_fy_end=is_fy_end,
            is_friday=is_friday,
            is_monday=is_monday,
            days_until_weekend=days_until_weekend,
            quarter_week=quarter_week,
            is_quarter_start=is_quarter_start,
            is_quarter_final_month=is_quarter_final_month,
            is_federal_holiday=False,  # Will be set by _mark_federal_holidays
        )

    def _mark_federal_holidays(self, calendar: List[FiscalDay]) -> None:
        """Mark federal holidays in calendar."""
        for fiscal_day in calendar:
            d = fiscal_day.date
            month = d.month
            day = d.day
            dow_num = fiscal_day.dow_num

            # New Year's Day
            if month == 1 and day == 1:
                fiscal_day.is_federal_holiday = True

            # Memorial Day (last Monday in May)
            elif month == 5 and dow_num == 2 and day > 24:
                fiscal_day.is_federal_holiday = True

            # Independence Day
            elif month == 7 and day == 4:
                fiscal_day.is_federal_holiday = True

            # Labor Day (first Monday in September)
            elif month == 9 and dow_num == 2 and day <= 7:
                fiscal_day.is_federal_holiday = True

            # Thanksgiving (fourth Thursday in November)
            elif month == 11 and dow_num == 5 and 22 <= day <= 28:
                fiscal_day.is_federal_holiday = True

            # Christmas
            elif month == 12 and day == 25:
                fiscal_day.is_federal_holiday = True

    # Lookup methods (match R interface)

    def get_day(self, sim_day: int) -> Optional[FiscalDay]:
        """Get fiscal day information for simulation day.

        Args:
            sim_day: Simulation day (1-indexed)

        Returns:
            FiscalDay object or None if out of range
        """
        if 1 <= sim_day <= len(self.calendar):
            return self.calendar[sim_day - 1]  # Convert to 0-indexed
        return None

    def get_current_quarter(self, sim_day: int) -> Optional[int]:
        """Get current fiscal quarter for simulation day.

        Args:
            sim_day: Simulation day (1-indexed)

        Returns:
            Fiscal quarter (1-4) or None if out of range
        """
        day = self.get_day(sim_day)
        return day.fiscal_quarter if day else None

    def is_new_quarter(self, sim_day: int) -> bool:
        """Check if simulation day is first day of new quarter.

        Args:
            sim_day: Simulation day (1-indexed)

        Returns:
            True if first day of quarter
        """
        if sim_day <= 1:
            return True  # Day 1 is always start of quarter
        if sim_day > len(self.calendar):
            return False

        current_day = self.get_day(sim_day)
        prev_day = self.get_day(sim_day - 1)

        if current_day and prev_day:
            return current_day.fiscal_quarter != prev_day.fiscal_quarter

        return False

    def is_new_fy(self, sim_day: int) -> bool:
        """Check if simulation day is first day of new fiscal year.

        Args:
            sim_day: Simulation day (1-indexed)

        Returns:
            True if first day of fiscal year
        """
        if sim_day <= 1:
            return True  # Day 1 is always start of FY
        if sim_day > len(self.calendar):
            return False

        current_day = self.get_day(sim_day)
        prev_day = self.get_day(sim_day - 1)

        if current_day and prev_day:
            return current_day.fiscal_year != prev_day.fiscal_year

        return False

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FiscalCalendar(start_fy={self.start_fy}, sim_days={self.sim_days}, "
            f"start_date={self.fy_start_date}, "
            f"end_date={self.metadata['end_date']})"
        )
