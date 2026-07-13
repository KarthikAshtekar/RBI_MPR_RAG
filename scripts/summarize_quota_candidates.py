import csv,json
from pathlib import Path
OUT=Path('reports/optimisation')
rows=[]
for name in ('QUOTA_EXPANDED','QUOTA_LARGE'):
 s=json.loads((OUT/name/'summary.json').read_text())
 rows.append({'experiment_id':name,'configuration':s['config'],
              'complete_evidence_recall':s['complete_evidence_recall'],
              'all_reports_hit':s['all_reports_hit'],'evidence_recall':s['evidence_recall'],
              'macro_mrr':s['macro_mrr'],'report_coverage':s['report_coverage'],
              'contamination':s['contamination'],'estimated_tokens':s['mean_estimated_tokens'],
              'selected_characters':s['mean_selected_characters'],
              'mean_latency':'unavailable','median_latency':'unavailable','p95_latency':'unavailable',
              'formal_selection_eligible':False,
              'ineligibility_reason':'Required per-question latency and selection traces are missing.'})
base=json.loads((OUT/'temporal_baseline'/'summary.json').read_text())
for r in rows:
 r['complete_evidence_gain_per_1000_tokens']=(r['complete_evidence_recall']-base['complete_evidence_recall'])/(r['estimated_tokens']/1000)
 r['evidence_recall_gain_per_1000_tokens']=(r['evidence_recall']-base['evidence_recall'])/(r['estimated_tokens']/1000)
(OUT/'quota_candidate_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
with (OUT/'quota_candidate_comparison.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,dict) else v for k,v in r.items()} for r in rows])
(OUT/'quota_candidate_comparison.md').write_text('# Quota Candidate Comparison\n\nBoth candidates remain ineligible for formal selection because latency traces are unavailable.\n\n'+ '\n'.join(f"- {r['experiment_id']}: CER={r['complete_evidence_recall']:.3f}, evidence recall={r['evidence_recall']:.3f}, tokens={r['estimated_tokens']:.0f}" for r in rows)+'\n')
(OUT/'stage_a_selection_status.json').write_text(json.dumps({'status':'invalidated_pending_selective_rerun','previous_provisional_selection':'QUOTA_LARGE','reason':'0 of 20 artifacts pass the strict integrity contract','heldout_accessed':False,'generation_executed':False,'groq_api_key_available':True},indent=2)+'\n')
print(json.dumps(rows,indent=2))
