from pathlib import Path
import pytest
from rbi_rag.report_registry import ReportRegistry


def write_registry(tmp_path, rows):
    import yaml
    path = tmp_path / "reports.yaml"
    path.write_text(yaml.safe_dump({"registry_version": "v1", "reports": rows}))
    return path


def row(report_id="a", date="2025-04-01", path="missing.pdf"):
    return {"report_id": report_id, "report_period": "April 2025", "report_date": date,
            "report_year": 2025, "report_month": 4, "report_type": "RBI_MPR",
            "pdf_path": path, "enabled": True}


def test_report_registry_validation_and_missing_pdf(tmp_path):
    registry = ReportRegistry.from_yaml(write_registry(tmp_path, [row()]))
    assert registry.version == "v1"
    assert registry.missing_paths() == ["missing.pdf"]


def test_duplicate_report_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        ReportRegistry.from_yaml(write_registry(tmp_path, [row(), row()]))


def test_invalid_report_dates_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid report_date"):
        ReportRegistry.from_yaml(write_registry(tmp_path, [row(date="not-a-date")]))

