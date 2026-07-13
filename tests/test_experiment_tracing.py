from langchain_core.documents import Document
from rbi_rag.experiment_tracing import LATENCY_FIELDS,StageTimer,context_statistics,validate_latency_schema

def test_complete_numeric_nonnegative_latency_schema():
    timer=StageTimer(); timer.measure('selection_latency_ms',lambda: 1); row=timer.finish()
    assert not validate_latency_schema(row) and row['total_retrieval_latency_ms']>0
    assert all(isinstance(row[x],float) and row[x]>=0 for x in LATENCY_FIELDS)

def test_missing_latency_is_rejected():
    assert validate_latency_schema({})

def test_context_statistics_are_deterministic():
    docs=[Document(page_content='abc',metadata={'report_id':'r','page':1}),Document(page_content='abc',metadata={'report_id':'r','page':1})]
    value=context_statistics(docs)
    assert value['selected_character_count']==6 and value['selected_chunk_count']==2
    assert value['unique_page_count']==1 and value['duplicate_chunk_count']==1
