from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import yaml


@dataclass(frozen=True)
class ReportSpec:
    report_id: str
    report_period: str
    report_date: date
    report_year: int
    report_month: int
    report_type: str
    pdf_path: Path
    enabled: bool

    @property
    def available(self) -> bool:
        return self.pdf_path.is_file()


@dataclass(frozen=True)
class ReportRegistry:
    version: str
    reports: tuple[ReportSpec, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> "ReportRegistry":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        reports = []
        for row in raw.get("reports", []):
            try:
                parsed_date = date.fromisoformat(str(row["report_date"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid report_date for {row.get('report_id')}") from exc
            if parsed_date.year != int(row["report_year"]) or parsed_date.month != int(row["report_month"]):
                raise ValueError(f"date fields disagree for {row['report_id']}")
            reports.append(ReportSpec(
                report_id=str(row["report_id"]), report_period=str(row["report_period"]),
                report_date=parsed_date, report_year=int(row["report_year"]),
                report_month=int(row["report_month"]), report_type=str(row["report_type"]),
                pdf_path=Path(row["pdf_path"]), enabled=bool(row.get("enabled", True)),
            ))
        ids = [report.report_id for report in reports]
        if not reports or len(ids) != len(set(ids)):
            raise ValueError("registry must be non-empty with unique report IDs")
        return cls(str(raw.get("registry_version", "unversioned")), tuple(reports))

    def enabled(self) -> tuple[ReportSpec, ...]:
        return tuple(sorted((r for r in self.reports if r.enabled), key=lambda r: r.report_date))

    def available(self) -> tuple[ReportSpec, ...]:
        return tuple(r for r in self.enabled() if r.available)

    def by_id(self) -> dict[str, ReportSpec]:
        return {r.report_id: r for r in self.reports}

    def missing_paths(self) -> list[str]:
        return [str(r.pdf_path) for r in self.enabled() if not r.available]

