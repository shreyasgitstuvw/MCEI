"""Unit tests for CorporateActionEventStudy."""
from datetime import date, timedelta

import pandas as pd
import numpy as np
import pytest

from src.analytics.corporate_actions import CorporateActionEventStudy, CorporateActionCalendar

BASE_DATE = date(2026, 2, 3)


def make_ca_df():
    """Corporate actions DataFrame with the columns the class actually reads."""
    return pd.DataFrame({
        "symbol": ["RELIANCE", "TCS"],
        "series": ["EQ", "EQ"],
        "security_name": ["Reliance Industries Ltd", "TCS Ltd"],
        "trade_date": [BASE_DATE, BASE_DATE],
        "ex_dt": [BASE_DATE + timedelta(days=5), BASE_DATE + timedelta(days=10)],
        "record_date": [BASE_DATE + timedelta(days=5), BASE_DATE + timedelta(days=10)],
        "purpose": ["DIVIDEND - RS 10 PER SHARE", "BONUS 1:1"],
        "action_type": ["DIVIDEND", "BONUS"],
    })


class TestCorporateActionEventStudy:

    def test_init_does_not_crash(self, price_df):
        ca = make_ca_df()
        study = CorporateActionEventStudy(ca, price_df)
        assert hasattr(study, "ca_data")
        assert hasattr(study, "price_data")

    def test_categorize_action(self):
        assert CorporateActionEventStudy._categorize_action("DIVIDEND - RS 10") == "Dividend"
        assert CorporateActionEventStudy._categorize_action("BONUS 1:1") == "Bonus"
        assert CorporateActionEventStudy._categorize_action("STOCK SPLIT 2:1") == "Stock Split"
        assert CorporateActionEventStudy._categorize_action("UNKNOWN") == "Other"

    def test_calculate_abnormal_returns_empty_when_no_overlap(self, price_df):
        # Use ex_dates far in the future so there's no overlapping price data
        ca = pd.DataFrame({
            "symbol": ["RELIANCE"],
            "purpose": ["DIVIDEND - RS 5"],
            "ex_dt": [date(2030, 1, 1)],
        })
        study = CorporateActionEventStudy(ca, price_df)
        result = study.calculate_abnormal_returns()
        assert isinstance(result, pd.DataFrame)

    def test_detect_announcement_leakage_no_crash_on_empty(self, price_df):
        # Previously crashed with KeyError: 'event_day' when CA had no price overlap
        ca = pd.DataFrame({
            "symbol": ["UNKNOWN_SYM"],
            "purpose": ["DIVIDEND - RS 5"],
            "ex_dt": [date(2030, 1, 1)],
        })
        study = CorporateActionEventStudy(ca, price_df)
        result = study.detect_announcement_leakage()
        assert isinstance(result, pd.DataFrame)

    def test_analyze_by_action_type_takes_abnormal_returns_df(self, price_df):
        # analyze_by_action_type requires an abnormal_returns DataFrame as argument
        ca = make_ca_df()
        study = CorporateActionEventStudy(ca, price_df)
        # Provide a minimal abnormal_returns DataFrame with required columns
        ar_df = pd.DataFrame({
            "action_type": pd.Series([], dtype=str),
            "event_day": pd.Series([], dtype=int),
            "abnormal_return": pd.Series([], dtype=float),
            "actual_return": pd.Series([], dtype=float),
        })
        result = study.analyze_by_action_type(ar_df)
        assert isinstance(result, pd.DataFrame)

    def test_analyze_dividend_impact_returns_dataframe(self, price_df):
        ca = make_ca_df()
        study = CorporateActionEventStudy(ca, price_df)
        result = study.analyze_dividend_impact()
        assert isinstance(result, pd.DataFrame)


class TestCorporateActionCalendar:

    def _make_calendar_df(self, ex_dt_value):
        # CorporateActionCalendar.get_upcoming_actions returns ['symbol','security','ex_dt','purpose']
        return pd.DataFrame({
            "symbol": ["RELIANCE"],
            "security": ["Reliance Industries"],   # 'security' not 'security_name'
            "ex_dt": [ex_dt_value],
            "purpose": ["DIVIDEND - RS 10"],
            "action_type": ["DIVIDEND"],
        })

    def test_get_upcoming_actions_returns_dataframe(self):
        from datetime import datetime, timedelta
        future = (datetime.now() + timedelta(days=5)).date()
        ca = self._make_calendar_df(future)
        cal = CorporateActionCalendar(ca)
        result = cal.get_upcoming_actions(days_ahead=30)
        assert isinstance(result, pd.DataFrame)

    def test_get_actions_by_date_range_returns_dataframe(self):
        from datetime import datetime
        ca = self._make_calendar_df(BASE_DATE + timedelta(days=5))
        cal = CorporateActionCalendar(ca)
        result = cal.get_actions_by_date_range(
            start_date=datetime(BASE_DATE.year, BASE_DATE.month, BASE_DATE.day),
            end_date=datetime(BASE_DATE.year, BASE_DATE.month, BASE_DATE.day) + timedelta(days=30),
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
