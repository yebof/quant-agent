import pandas as pd
from fredapi import Fred


class MacroDataProvider:
    def __init__(self, api_key: str):
        self.fred = Fred(api_key=api_key)

    def get_vix(self, lookback_days: int = 30) -> dict:
        series = self.fred.get_series(
            "VIXCLS",
            observation_start=pd.Timestamp.now() - pd.Timedelta(days=lookback_days),
        )
        series = series.dropna()
        if series.empty:
            return {"current": None, "mean_5d": None, "trend": "unknown"}
        current = float(series.iloc[-1])
        mean_5d = float(series.tail(5).mean())
        if len(series) >= 5:
            prev = float(series.iloc[-5])
            trend = "rising" if current > prev else "falling" if current < prev else "flat"
        else:
            trend = "unknown"
        return {"current": current, "mean_5d": mean_5d, "trend": trend}

    def get_treasury_yields(self) -> dict:
        us2y_series = self.fred.get_series(
            "DGS2",
            observation_start=pd.Timestamp.now() - pd.Timedelta(days=7),
        )
        us10y_series = self.fred.get_series(
            "DGS10",
            observation_start=pd.Timestamp.now() - pd.Timedelta(days=7),
        )
        us2y = float(us2y_series.dropna().iloc[-1]) if not us2y_series.dropna().empty else None
        us10y = float(us10y_series.dropna().iloc[-1]) if not us10y_series.dropna().empty else None
        spread = (us10y - us2y) if us2y and us10y else None
        return {
            "us2y": us2y,
            "us10y": us10y,
            "spread_2_10": round(spread, 4) if spread is not None else None,
            "inverted": spread < 0 if spread is not None else None,
        }

    def get_fed_funds_rate(self) -> float | None:
        series = self.fred.get_series(
            "FEDFUNDS",
            observation_start=pd.Timestamp.now() - pd.Timedelta(days=60),
        )
        series = series.dropna()
        return float(series.iloc[-1]) if not series.empty else None

    def get_macro_summary(self) -> dict:
        return {
            "vix": self.get_vix(),
            "treasury": self.get_treasury_yields(),
            "fed_funds_rate": self.get_fed_funds_rate(),
        }
