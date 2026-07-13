from datetime import date
from types import SimpleNamespace
from rbi_rag.multi_config import MultiReportConfig
from rbi_rag.multi_index import ingest_report
from rbi_rag.report_registry import ReportSpec


class Doc:
    def __init__(self, text, page): self.page_content = text; self.metadata = {"page": page}


def test_stable_complete_chunk_metadata(monkeypatch, tmp_path):
    pdf = tmp_path / "report.pdf"; pdf.write_bytes(b"pdf")
    monkeypatch.setattr("rbi_rag.multi_index.PyPDFLoader", lambda path: SimpleNamespace(load=lambda: [Doc("Monetary Policy Report April 2025", 0)]))
    monkeypatch.setattr("rbi_rag.multi_index.split_pages", lambda pages, **kwargs: [Doc("a", 0), Doc("b", 0)])
    config = MultiReportConfig.from_yaml(__import__("pathlib").Path("configs/multi_report.yaml"))
    report = ReportSpec("rbi_mpr_2025_04", "April 2025", date(2025,4,1), 2025, 4, "RBI_MPR", pdf, True)
    _, chunks = ingest_report(report, config)
    assert chunks[0].metadata["chunk_id"] == "rbi_mpr_2025_04_p001_c000"
    assert chunks[1].metadata["chunk_id"] == "rbi_mpr_2025_04_p001_c001"
    assert chunks[0].metadata["page_index"] == 0 and chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["section"] is None
