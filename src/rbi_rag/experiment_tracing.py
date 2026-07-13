from __future__ import annotations
from contextlib import contextmanager
from math import ceil
import time

LATENCY_FIELDS=('routing_latency_ms','query_transformation_latency_ms','dense_latency_ms',
'bm25_latency_ms','candidate_union_latency_ms','fusion_latency_ms','reranking_latency_ms',
'selection_latency_ms','deduplication_latency_ms','context_construction_latency_ms',
'total_retrieval_latency_ms')

LOSS_STAGES=(
    'not_in_dense_candidates','not_in_bm25_candidates','not_in_candidate_union',
    'lost_in_fusion','lost_before_reranker','lost_in_reranking','lost_by_quota',
    'lost_by_deduplication','extraction_does_not_preserve_evidence',
    'annotation_mismatch','evidence_found','unknown',
)

def context_statistics(chunks):
    normalised=[' '.join(c.page_content.split()) for c in chunks]
    chars=sum(len(c.page_content) for c in chunks)
    pages={(c.metadata.get('report_id'),c.metadata.get('page')) for c in chunks}
    duplicates=len(normalised)-len(set(normalised))
    seen=set(); repeated=0
    for value in normalised:
        if value in seen:
            repeated += len(value)
        seen.add(value)
    return {'selected_character_count':chars,'estimated_token_count':ceil(chars/4) if chars else 0,
            'selected_chunk_count':len(chunks),'unique_page_count':len(pages),
            'duplicate_chunk_count':duplicates,
            'repeated_text_ratio':min(1.0,max(0.0,repeated/chars if chars else 0.0))}

def validate_latency_schema(row):
    issues=[]
    for field in LATENCY_FIELDS:
        value=row.get(field)
        if not isinstance(value,(int,float)): issues.append('missing_or_non_numeric:'+field)
        elif value<0: issues.append('negative:'+field)
    if isinstance(row.get('total_retrieval_latency_ms'),(int,float)) and row['total_retrieval_latency_ms']<=0:
        issues.append('non_positive_total')
    return issues

def first_evidence_rank(pages, accepted_pages):
    accepted=set(accepted_pages or [])
    return next((rank for rank,page in enumerate(pages,1) if page in accepted),None)

def recompute_loss_stage(trace):
    if trace.get('accepted_evidence_found'):
        return 'evidence_found'
    if trace.get('annotation_mismatch'):
        return 'annotation_mismatch'
    dense=trace.get('dense_first_evidence_rank') is not None
    bm25=trace.get('bm25_first_evidence_rank') is not None
    union=any(page in set(trace.get('accepted_pages') or []) for page in trace.get('candidate_union_pages') or [])
    rrf=trace.get('rrf_first_evidence_rank') is not None
    reranker_in=any(page in set(trace.get('accepted_pages') or []) for page in trace.get('reranker_input_pages') or [])
    reranker_out=trace.get('evidence_rank_after_reranking') is not None
    before=set(trace.get('selected_chunk_ids_before_dedup') or [])
    after=set(trace.get('selected_chunk_ids_after_dedup') or [])
    if not dense and not bm25:
        return 'not_in_candidate_union'
    if not union:
        return 'not_in_candidate_union'
    if not rrf:
        return 'lost_in_fusion'
    if not reranker_in:
        return 'lost_before_reranker'
    if not reranker_out:
        return 'lost_in_reranking'
    if before and before != after and not trace.get('accepted_evidence_found'):
        return 'lost_by_deduplication'
    return 'lost_by_quota'

class StageTimer:
    def __init__(self):
        self.started=time.perf_counter()
        self.values={field:0.0 for field in LATENCY_FIELDS}

    def measure(self,field,fn=None):
        if fn is not None:
            return self.call(field,fn)

        @contextmanager
        def _measurement():
            start=time.perf_counter()
            try:
                yield
            finally:
                self.values[field]=(time.perf_counter()-start)*1000
        return _measurement()

    def call(self,field,fn):
        start=time.perf_counter()
        result=fn()
        self.values[field]=(time.perf_counter()-start)*1000
        return result

    def finish(self):
        self.values['total_retrieval_latency_ms']=(time.perf_counter()-self.started)*1000
        return {key:float(value) for key,value in self.values.items()}
