"""Token-based maintenance resource allocation tracker.

ARCHITECTURE:
- Single token pool (not separate pools by type)
- Weighted consumption by maintenance type:
  * Preventive/Reactive: 1 token
  * Minor Phase: 3 tokens
  * Major Phase: 10 tokens
- Quarterly allocation: annual_budget / 4 (remainder goes to Q1)
- Tokens accumulate across quarters (don't expire)
- Example: 120 tokens → Q1: 30, Q2: 30, Q3: 30, Q4: 30
- Example: 122 tokens → Q1: 31, Q2: 30, Q3: 30, Q4: 31
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .fiscal_calendar import FiscalCalendar

# Token weight constants (cost per maintenance type)
TOKEN_WEIGHTS = {
    "preventive": 1,
    "reactive": 1,
    "minor_phase": 3,
    "major_phase": 10,
}


@dataclass
class TokenTransaction:
    """Record of a single token consumption event."""

    sim_day: int
    maintenance_type: str  # 'preventive', 'reactive', 'minor_phase', 'major_phase'
    aircraft_id: int
    description: str
    fiscal_year: int
    quarter: int


@dataclass
class QuarterData:
    """Tracking data for a single quarter."""

    allocated_tokens: int = 0  # Tokens allocated this quarter
    released: bool = False  # Whether quarter tokens have been released
    used_tokens: int = 0  # Tokens consumed this quarter


class TokenTracker:
    """Token-based maintenance resource allocation tracker.

    Manages quarterly token allocation and consumption with accumulation.
    """

    def __init__(self, annual_token_budget: int, fiscal_calendar: FiscalCalendar, verbose: bool = False):
        """Initialize token tracker.

        Args:
            annual_token_budget: Total annual tokens (e.g., 120)
            fiscal_calendar: FiscalCalendar instance for FY/quarter tracking
            verbose: Enable detailed logging
        """
        # Configuration
        self.annual_token_budget = annual_token_budget
        self.fiscal_calendar = fiscal_calendar
        self.verbose = verbose
        self.current_fy = fiscal_calendar.start_fy

        # Token availability (single pool, accumulates across quarters)
        self.tokens_available = 0

        # Annual totals
        self.total_allocated = 0  # Total tokens allocated this FY
        self.total_used = 0  # Total tokens consumed this FY
        self.total_unreleased = annual_token_budget  # Tokens not yet released

        # Quarterly tracking structure
        # quarters_by_fy[fiscal_year][quarter] -> QuarterData
        self.quarters_by_fy: dict[int, dict[int, QuarterData]] = {}

        # Initialize first fiscal year
        self._initialize_fy_structure(self.current_fy)

        # Consumption history
        self.transactions: List[TokenTransaction] = []
        self.daily_usage: List[int] = []  # Tokens consumed per day (indexed by sim_day - 1)

    def _initialize_fy_structure(self, fy: int) -> None:
        """Initialize quarterly tracking structure for a fiscal year.

        Args:
            fy: Fiscal year to initialize
        """
        # Fiscal year transition: reset for new year
        if fy > self.current_fy:
            if self.verbose:
                print(f"🗓️  Fiscal year transition: FY{self.current_fy} -> FY{fy}")

            # Reset token availability
            self.tokens_available = 0

            # Reset totals
            self.total_allocated = 0
            self.total_used = 0
            self.total_unreleased = self.annual_token_budget

        # Initialize quarters structure
        self.quarters_by_fy[fy] = {q: QuarterData() for q in range(1, 5)}

    def get_current_fy(self, sim_day: int) -> int:
        """Get current fiscal year for simulation day.

        Args:
            sim_day: Simulation day (1-indexed)

        Returns:
            Fiscal year (e.g., 2025)
        """
        if sim_day <= 0:
            return self.current_fy

        fiscal_day = self.fiscal_calendar.get_day(sim_day)
        if fiscal_day:
            return fiscal_day.fiscal_year

        # Fallback: estimate based on days
        return self.fiscal_calendar.start_fy + (sim_day // 365)

    def check_token_available(self, maintenance_type: str) -> bool:
        """Check if tokens are available for given maintenance type.

        Args:
            maintenance_type: Type of maintenance ('preventive', 'reactive', 'minor_phase', 'major_phase')

        Returns:
            True if sufficient tokens available for this maintenance type
        """
        required_tokens = TOKEN_WEIGHTS.get(maintenance_type, 1)
        return self.tokens_available >= required_tokens

    def consume_token(
        self, maintenance_type: str, aircraft_id: int, sim_day: int, description: str = ""
    ) -> bool:
        """Consume tokens based on maintenance type weight.

        Args:
            maintenance_type: Type of maintenance ('preventive', 'reactive', 'minor_phase', 'major_phase')
            aircraft_id: Aircraft consuming the token
            sim_day: Current simulation day
            description: Optional description of consumption

        Returns:
            True if consumption successful, False if insufficient tokens
        """
        # Get cost for this maintenance type
        token_cost = TOKEN_WEIGHTS.get(maintenance_type, 1)

        # Check availability
        if not self.check_token_available(maintenance_type):
            return False

        # Consume tokens (weighted by maintenance type)
        self.tokens_available -= token_cost
        self.total_used += token_cost

        # Update quarterly usage tracking
        current_fy = self.get_current_fy(sim_day)
        fiscal_day = self.fiscal_calendar.get_day(sim_day)
        current_q = fiscal_day.fiscal_quarter if fiscal_day else None

        if current_q and 1 <= current_q <= 4:
            # Ensure fiscal year structure exists
            if current_fy not in self.quarters_by_fy:
                self._initialize_fy_structure(current_fy)

            # Update quarterly usage (weighted by token cost)
            self.quarters_by_fy[current_fy][current_q].used_tokens += token_cost

        # Record transaction
        self.transactions.append(
            TokenTransaction(
                sim_day=sim_day,
                maintenance_type=maintenance_type,
                aircraft_id=aircraft_id,
                description=description,
                fiscal_year=current_fy,
                quarter=current_q if current_q else 0,
            )
        )

        # Update daily usage tracking (weighted by token cost)
        day_idx = sim_day - 1  # Convert to 0-indexed
        while len(self.daily_usage) <= day_idx:
            self.daily_usage.append(0)
        self.daily_usage[day_idx] += token_cost

        if self.verbose:
            print(
                f"💸 Tokens consumed: {token_cost} for {maintenance_type} maintenance on aircraft {aircraft_id} "
                f"(remaining: {self.tokens_available})"
            )

        return True

    def get_quarterly_allocation(self, quarter: int) -> int:
        """Calculate token allocation for a specific quarter.

        Args:
            quarter: Quarter number (1-4)

        Returns:
            Number of tokens to allocate this quarter
        """
        base_allocation = self.annual_token_budget // 4
        remainder = self.annual_token_budget % 4

        # Q1 gets base + remainder, Q2-Q4 get base
        if quarter == 1:
            return base_allocation + remainder
        else:
            return base_allocation

    def release_quarterly_tokens(self, quarter: int, sim_day: Optional[int] = None) -> Dict:
        """Release quarterly token allocation.

        Args:
            quarter: Quarter to release (1-4)
            sim_day: Current simulation day (for FY tracking)

        Returns:
            Dict with 'success' (bool) and optional 'reason', 'tokens_released'
        """
        if quarter < 1 or quarter > 4:
            return {"success": False, "reason": "invalid_quarter"}

        # Get current fiscal year
        current_fy = self.current_fy if sim_day is None else self.get_current_fy(sim_day)

        # Ensure fiscal year structure exists
        if current_fy not in self.quarters_by_fy:
            self._initialize_fy_structure(current_fy)
            self.total_unreleased = self.annual_token_budget

        # Check if already released
        if self.quarters_by_fy[current_fy][quarter].released:
            return {"success": False, "reason": "already_released"}

        # Calculate quarterly allocation
        tokens_to_release = self.get_quarterly_allocation(quarter)

        # Check annual budget constraint
        if self.total_allocated + tokens_to_release > self.annual_token_budget:
            if self.verbose:
                print(
                    f"⚠️  Annual budget exceeded: Cannot allocate {tokens_to_release} tokens "
                    f"(would exceed {self.annual_token_budget} budget)"
                )
            return {"success": False, "reason": "annual_budget_exceeded"}

        # Execute the allocation
        self.quarters_by_fy[current_fy][quarter].allocated_tokens = tokens_to_release
        self.quarters_by_fy[current_fy][quarter].released = True

        # Add tokens to available pool (accumulates)
        self.tokens_available += tokens_to_release

        # Update totals
        self.total_allocated += tokens_to_release
        self.total_unreleased -= tokens_to_release
        self.current_fy = current_fy

        if self.verbose:
            print(
                f"💰 Released Q{quarter} tokens: {tokens_to_release} "
                f"(total available: {self.tokens_available})"
            )

        return {"success": True, "tokens_released": tokens_to_release}

    def get_usage_rate(self, window: int = 30) -> float:
        """Calculate recent token usage rate (for decision making).

        Args:
            window: Number of days to look back

        Returns:
            Average daily token consumption
        """
        if not self.daily_usage:
            return 0.0

        window = min(window, len(self.daily_usage))
        if window > 0:
            recent_usage = self.daily_usage[-window:]
            return sum(recent_usage) / len(recent_usage)

        return 0.0

    def get_summary(self, sim_day: Optional[int] = None) -> Dict:
        """Get comprehensive summary for decision making.

        Args:
            sim_day: Current simulation day

        Returns:
            Summary dict with budget, allocation, usage, and utilization info
        """
        current_fy = self.current_fy if sim_day is None else self.get_current_fy(sim_day)
        usage_rate = self.get_usage_rate()

        # Calculate utilization rate
        utilization_rate = self.total_used / self.total_allocated if self.total_allocated > 0 else 0.0

        # Count quarters released this fiscal year
        quarters_released = 0
        if current_fy in self.quarters_by_fy:
            quarters_released = sum(
                1 for q in range(1, 5) if self.quarters_by_fy[current_fy][q].released
            )

        return {
            "annual_token_budget": self.annual_token_budget,
            "total_tokens_allocated": self.total_allocated,
            "total_tokens_used": self.total_used,
            "total_tokens_unreleased": self.total_unreleased,
            "tokens_available": self.tokens_available,
            "utilization_rate": utilization_rate,
            "usage_rate_daily": usage_rate,
            "quarters_released": quarters_released,
            "current_fiscal_year": current_fy,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TokenTracker(annual_budget={self.annual_token_budget}, "
            f"allocated={self.total_allocated}, used={self.total_used}, "
            f"available={self.tokens_available})"
        )
