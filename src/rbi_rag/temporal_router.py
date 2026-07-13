from __future__ import annotations

import re
from .report_registry import ReportRegistry
from .schemas import QueryPlan

MONTHS = {"april": 4, "apr": 4, "october": 10, "oct": 10}
TREND_WORDS = ("change", "changed", "evolve", "evolved", "over time", "trend", "across reports", "all reports")
COMPARE_WORDS = ("compare", "versus", " vs ", "difference between")


class TemporalQueryRouter:
    def __init__(self, registry: ReportRegistry):
        self.registry = registry

    def route(self, query: str) -> QueryPlan:
        normalized = " ".join(query.lower().split())
        enabled = self.registry.enabled()
        by_period = {(r.report_year, r.report_month): r for r in enabled}
        mentioned = self._mentioned_periods(normalized)
        if "six months later" in normalized and len(mentioned) == 1:
            year, month = mentioned[0]
            later = (year + (month + 6 - 1) // 12, (month + 6 - 1) % 12 + 1)
            mentioned.append(later)
        unsupported = [period for period in mentioned if period not in by_period]
        supported = sorted({by_period[p] for p in mentioned if p in by_period}, key=lambda r: r.report_date)
        topic = next((word for word in ("inflation", "growth", "liquidity", "repo rate", "policy stance")
                      if word in normalized), None)
        calculation = any(word in normalized for word in ("difference", "by how much", "subtract", "change in"))
        if unsupported:
            labels = ", ".join(f"{month}/{year}" for year, month in unsupported)
            return self._plan(query, normalized, "unsupported_period", supported, topic,
                              calculation, .99, f"Unavailable requested period(s): {labels}")
        if "latest" in normalized:
            return self._plan(query, normalized, "latest_report", [enabled[-1]], topic,
                              calculation, .99, "Latest chronological registered report requested")
        if "earliest" in normalized:
            return self._plan(query, normalized, "single_report", [enabled[0]], topic,
                              calculation, .99, "Earliest chronological registered report requested")
        if "previous" in normalized:
            target = enabled[-2] if len(enabled) > 1 else enabled[-1]
            return self._plan(query, normalized, "single_report", [target], topic,
                              calculation, .9, "Previous report interpreted relative to latest registered report")
        is_trend = any(word in normalized for word in TREND_WORDS)
        if is_trend and not mentioned:
            return self._plan(query, normalized, "trend_all_reports", enabled, topic,
                              calculation, .95, "Trend/all-report keyword detected")
        if len(supported) >= 3 or (is_trend and len(supported) > 1):
            return self._plan(query, normalized, "trend_all_reports", supported or enabled,
                              topic, calculation, .98, "Trend across multiple explicit periods")
        if len(supported) == 2 or (len(supported) > 1 and any(w in f" {normalized} " for w in COMPARE_WORDS)):
            return self._plan(query, normalized, "pairwise_comparison", supported, topic,
                              calculation, .99, "Two registered report periods requested")
        if len(supported) == 1:
            return self._plan(query, normalized, "single_report", supported, topic,
                              calculation, .99, "Exact registered report period matched")
        return self._plan(query, normalized, "global_unspecified", enabled, topic,
                          calculation, .7, "No explicit period; search all reports with report balance")

    def _mentioned_periods(self, text: str):
        periods = []
        years = [int(value) for value in re.findall(r"\b20\d{2}\b", text)]
        for month_name, month in MONTHS.items():
            for match in re.finditer(rf"\b{month_name}\b", text):
                nearby = re.search(r"\b(20\d{2})\b", text[match.end():match.end() + 12])
                before = re.findall(r"\b20\d{2}\b", text[max(0, match.start() - 12):match.start()])
                year = int(nearby.group(1)) if nearby else (int(before[-1]) if before else (years[0] if len(set(years)) == 1 else None))
                if year is not None:
                    periods.append((year, month))
        return list(dict.fromkeys(periods))

    @staticmethod
    def _plan(query, normalized, kind, reports, topic, calculation, confidence, reason):
        return QueryPlan(query, normalized, kind, tuple(r.report_id for r in reports), topic,
                         "temporal" if kind in ("pairwise_comparison", "trend_all_reports") else None,
                         calculation, confidence, reason)
