from langchain_core.documents import Document
from rbi_rag.comparative_generation import citations_from_context, format_source_context, validate_citations
from rbi_rag.multi_metrics import evaluate_multi_retrieval
from rbi_rag.report_registry import ReportRegistry
from pathlib import Path


def chunk(report, period, page, cid):
    return Document(page_content="inflation evidence", metadata={"report_id":report,"report_period":period,"page":page,"chunk_id":cid})


def test_report_coverage_all_reports_hit_and_per_report_mrr():
    a=chunk("a","April",2,"a1"); b=chunk("b","October",5,"b1")
    case={"question_id":"q","query_type":"pairwise_comparison","required_report_ids":["a","b"],
          "ground_truth":{"a":{"accepted_pages":[2],"expected_evidence":[]},"b":{"accepted_pages":[5],"expected_evidence":[]}}}
    result={"final_selected_chunks":[a,b],"final_chunk_quota_by_report":{"a":1,"b":1},"missing_report_warnings":[]}
    metrics=evaluate_multi_retrieval(case,result)
    assert metrics["report_coverage"]==1 and metrics["all_reports_hit"] is True
    assert metrics["per_report_mrr"]=={"a":1.0,"b":1.0} and metrics["macro_report_mrr"]==1.0


def test_source_labels_and_citation_validation():
    registry=ReportRegistry.from_yaml(Path("configs/reports.yaml"))
    value=chunk("rbi_mpr_2025_04","April 2025",2,"a1")
    context=format_source_context([value],registry)
    citations=citations_from_context([value])
    assert '<SOURCE' in context and 'report_period="April 2025"' in context
    assert validate_citations(citations,[value])
    bad=citations[0].__class__(citations[0].report_id,citations[0].report_period,2,"other","")
    assert not validate_citations([bad],[value])

