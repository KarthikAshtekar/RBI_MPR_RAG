from __future__ import annotations
import csv,json
from collections import Counter
from hashlib import sha256
from pathlib import Path

root=Path('.')
out=root/'reports/optimisation'; out.mkdir(parents=True,exist_ok=True)
protected=[root/'reports/current/baseline_report.md',root/'reports/current/retrieval_raw_results.json',root/'reports/current/retrieval_question_results.csv',root/'reports/current/retrieval_pipeline_summary.csv',root/'reports/current/retrieval_summary.json',root/'reports/multi_report/full_temporal_retrieval_report.md',root/'reports/multi_report/architecture_comparison.json',root/'reports/multi_report/architecture_comparison.csv',root/'reports/multi_report/extraction_audit.json',root/'reports/multi_report/extraction_audit.md',root/'reports/multi_report/dev_failure_analysis.csv',root/'reports/multi_report/dev_failure_analysis.md',root/'data/evaluation/temporal_dataset_manifest.json']
checks={str(p):sha256(p.read_bytes()).hexdigest() for p in protected}
(out/'pre_optimisation_checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
cases={x['question_id']:x for x in map(json.loads,(root/'data/evaluation/multi_report_dev.jsonl').read_text(encoding='utf-8').splitlines()) if x.get('verification_status')=='verified'}
raw=json.loads((root/'reports/multi_report/retrieval_dev_raw_results.json').read_text())
rows=[]
for q in raw['rows']:
 case=cases[q['question_id']]; trace=q['retrieval_trace']
 for rid in case['required_report_ids']:
  accepted=set(case['ground_truth'][rid]['accepted_pages']); expected=case['ground_truth'][rid]['expected_evidence']
  def pages(stage): return [x['page'] for x in trace[stage].get(rid,[])]
  dense,bm25,rrf,reranked=pages('dense_candidates_by_report'),pages('bm25_candidates_by_report'),pages('rrf_candidates_by_report'),pages('reranked_candidates_by_report')
  selected=[x['page'] for x in trace['selected_chunks'] if x['report_id']==rid]
  rank=lambda xs: next((i for i,p in enumerate(xs,1) if p in accepted),None)
  dr,br,rr,rer=rank(dense),rank(bm25),rank(rrf),rank(reranked); sr=rank(selected)
  if sr: loss='evidence_found'
  elif not dr and not br: loss='not_in_candidate_union'
  elif not rr: loss='lost_in_fusion'
  elif not rer: loss='lost_in_reranking'
  else: loss='lost_by_quota'
  rows.append({'question_id':q['question_id'],'report_id':rid,'accepted_pages':sorted(accepted),'expected_evidence':expected,'dense_evidence_found':bool(dr),'dense_first_evidence_rank':dr,'bm25_evidence_found':bool(br),'bm25_first_evidence_rank':br,'union_evidence_found':bool(dr or br),'rrf_evidence_found':bool(rr),'rrf_first_evidence_rank':rr,'reranker_input_evidence_found':bool(rr),'reranker_output_evidence_rank':rer,'selected_before_dedup':bool(sr),'selected_after_dedup':bool(sr),'loss_stage':loss})
def recall(key): return sum(bool(r[key]) for r in rows)/len(rows)
summary={'required_report_cases':len(rows),'dense_candidate_recall_at_15':recall('dense_evidence_found'),'bm25_candidate_recall_at_15':recall('bm25_evidence_found'),'candidate_union_recall_at_15':recall('union_evidence_found'),'rrf_candidate_recall_at_15':recall('rrf_evidence_found'),'reranker_input_recall':recall('reranker_input_evidence_found'),'reranker_top_k_recall':recall('selected_after_dedup'),'loss_stage_counts':dict(Counter(r['loss_stage'] for r in rows))}
with (out/'baseline_stage_diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'baseline_stage_diagnostics.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2)+'\n')
(out/'baseline_stage_diagnostics.md').write_text('# Baseline Stage Diagnostics\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
print(json.dumps(summary,indent=2))
