from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

from pypdf import PdfReader

FILES = {
    "rbi_mpr_2025_04": Path("mpr_april_2025.pdf"),
    "rbi_mpr_2025_10": Path("data/raw/Oct_2025_RBI_MPR.pdf"),
    "rbi_mpr_2026_04": Path("data/raw/April_2026_RBI_MPR.pdf"),
}
PERIODS = {"rbi_mpr_2025_04": "April 2025", "rbi_mpr_2025_10": "October 2025", "rbi_mpr_2026_04": "April 2026"}

# Every excerpt below is asserted against normalized text from the cited PDF page before output.
FACTS = {
"a_food": ("rbi_mpr_2025_04",12,"food inflation which remained elevated at an average of 8.5 per cent during October- December 2024, decelerated to 3.8 per cent in February 2025","food_inflation","narrative"),
"a_repo": ("rbi_mpr_2025_04",12,"cut the policy repo rate by 25 basis points (bps) to 6.25 per cent","monetary_policy","narrative"),
"a_crude": ("rbi_mpr_2025_04",14,"US$ 70 per barrel during 2025-26","commodity_assumptions","table"),
"a_global": ("rbi_mpr_2025_04",14,"3.1 per cent in 2025 3.0 per cent in 2026","global_growth","table"),
"a_borrow": ("rbi_mpr_2025_04",57,"gross market borrowings amounted to ₹14.01 lakh crore","fiscal_borrowing","narrative"),
"a_gdp": ("rbi_mpr_2025_04",13,"Real GDP growth for 2025-26 was projected at 6.7 per cent","growth_outlook","narrative"),
"a_cpi": ("rbi_mpr_2025_04",13,"CPI inflation for 2025- 26 is projected at 4.2 per cent","inflation_outlook","narrative"),
"a_stance": ("rbi_mpr_2025_04",13,"continue with the neutral stance","monetary_policy","narrative"),
"o_food": ("rbi_mpr_2025_10",10,"food inflation turned negative in June and July 2025","food_inflation","narrative"),
"o_repo": ("rbi_mpr_2025_10",10,"bringing the repo rate down to 5.5 per cent","monetary_policy","narrative"),
"o_core": ("rbi_mpr_2025_10",10,"Core inflation ( i.e., CPI excluding food and fuel), however, largely remained steady around 4 per cent","core_inflation","narrative"),
"o_crude": ("rbi_mpr_2025_10",12,"US$ 70 per barrel during H2: 2025-26","commodity_assumptions","table"),
"o_global": ("rbi_mpr_2025_10",12,"3.0 per cent in 2025 3.1 per cent in 2026","global_growth","table"),
"o_borrow": ("rbi_mpr_2025_10",52,"gross market borrowings raised by the centre stood at ₹7.95 lakh crore","fiscal_borrowing","narrative"),
"o_stance": ("rbi_mpr_2025_10",11,"keep the repo rate unchanged at 5.5 per cent and to maintain the neutral stance","monetary_policy","narrative"),
"p_food": ("rbi_mpr_2026_04",5,"food group where the waning of base effects led to a turnaround from deflation","food_inflation","narrative"),
"p_repo": ("rbi_mpr_2026_04",26,"reduced the policy repo rate by 25 bps in the December meeting","monetary_policy","narrative"),
"p_core": ("rbi_mpr_2026_04",5,"core inflation remained contained barring precious metals","core_inflation","narrative"),
"p_crude": ("rbi_mpr_2026_04",5,"surge in global crude oil prices since the West Asia conflict","commodity_assumptions","chart_manually_verified"),
"p_global": ("rbi_mpr_2026_04",5,"Global growth remained resilient but below its historical average","global_growth","narrative"),
"p_borrow": ("rbi_mpr_2026_04",64,"gross market borrowings through issuance of dated securities amounted to ₹14.61 lakh crore","fiscal_borrowing","narrative"),
"p_gdp": ("rbi_mpr_2026_04",101,"real GDP growth for 2025–26 is estimated at 7.6 per cent","growth_outlook","narrative"),
"p_cpi": ("rbi_mpr_2026_04",103,"CPI inflation for 2025-26 was revised slightly upwards to 2.1 per cent","inflation_outlook","narrative"),
"p_stance": ("rbi_mpr_2026_04",26,"maintain the neutral stance adopted in June 2025","monetary_policy","narrative"),
}

def norm(value): return " ".join(value.split())

PAGES = {rid: [norm(page.extract_text() or "") for page in PdfReader(path).pages] for rid,path in FILES.items()}
for key,(rid,page,evidence,*_) in FACTS.items():
    if norm(evidence) not in PAGES[rid][page-1]:
        raise RuntimeError(f"Unverified evidence {key} on {rid} page {page}")

def case(case_id, question, query_type, fact_keys, split, expected=None):
    ground, categories, types = {}, [], []
    for key in fact_keys:
        rid,page,evidence,category,source_type=FACTS[key]
        entry=ground.setdefault(rid,{"accepted_pages":[],"expected_evidence":[],"evidence_excerpts":[]})
        entry["accepted_pages"].append(page); entry["expected_evidence"].append(norm(evidence)); entry["evidence_excerpts"].append(norm(evidence))
        categories.append(category); types.append(source_type)
    required=sorted(ground, key=lambda rid: (FILES_ORDER[rid]))
    answer=expected or " ".join(f"{PERIODS[rid]}: " + "; ".join(ground[rid]["expected_evidence"]) for rid in required)
    return {"question_id":case_id,"question":question,"query_type":query_type,
            "required_report_ids":required,"ground_truth":ground,"expected_answer":answer,
            "category":categories[0] if len(set(categories))==1 else "multi_topic",
            "source_information_type":sorted(set(types)),"split":split,
            "verification_status":"verified","verified_against":[f"{rid}_p{page:03d}" for rid,v in ground.items() for page in v["accepted_pages"]],
            "notes":"Evidence excerpts verified by exact normalized match against local PDF text."}

FILES_ORDER={"rbi_mpr_2025_04":0,"rbi_mpr_2025_10":1,"rbi_mpr_2026_04":2}

dev_specs=[
("oct_dev_001","What happened to food inflation in June and July 2025 according to the October 2025 MPR?","single_report",["o_food"]),
("oct_dev_002","To what level had the repo rate been brought by October 2025?","single_report",["o_repo"]),
("oct_dev_003","Where did core inflation broadly remain in the October 2025 report?","single_report",["o_core"]),
("oct_dev_004","What crude-oil baseline assumption was used for H2:2025-26 in October 2025?","single_report",["o_crude"]),
("oct_dev_005","What global-growth assumptions for 2025 and 2026 were used in October 2025?","single_report",["o_global"]),
("oct_dev_006","According to the October 2025 MPR, how much had the Centre raised through gross market borrowings by late September 2025?","single_report",["o_borrow"]),
("apr26_dev_001","What did April 2026 say about the food group's movement from deflation?","single_report",["p_food"]),
("apr26_dev_002","What repo-rate action in December was reported in April 2026?","single_report",["p_repo"]),
("apr26_dev_003","How did April 2026 characterise core inflation apart from precious metals?","single_report",["p_core"]),
("apr26_dev_004","What crude-oil risk arising from the West Asia conflict was highlighted in April 2026?","single_report",["p_crude"]),
("apr26_dev_005","How did April 2026 characterise global growth relative to its historical average?","single_report",["p_global"]),
("apr26_dev_006","What were the Centre's gross market borrowings in 2025-26 according to April 2026?","single_report",["p_borrow"]),
("cmp_dev_001","Compare food-inflation developments reported in April and October 2025.","pairwise_comparison",["a_food","o_food"]),
("cmp_dev_002","Compare the policy-rate levels described in April and October 2025.","pairwise_comparison",["a_repo","o_repo"]),
("cmp_dev_003","Compare the crude-oil baseline assumptions in April and October 2025.","pairwise_comparison",["a_crude","o_crude"]),
("cmp_dev_004","Compare global-growth assumptions in April and October 2025.","pairwise_comparison",["a_global","o_global"]),
("cmp_dev_005","Compare food-inflation language in October 2025 and April 2026.","pairwise_comparison",["o_food","p_food"]),
("cmp_dev_006","Compare the monetary-policy actions described in October 2025 and April 2026.","pairwise_comparison",["o_repo","p_repo"]),
("cmp_dev_007","Compare global-growth language in October 2025 and April 2026.","pairwise_comparison",["o_global","p_global"]),
("cmp_dev_008","Compare the Centre's borrowing figures in October 2025 and April 2026.","pairwise_comparison",["o_borrow","p_borrow"]),
("cmp_dev_009","Compare growth figures reported in April 2025 and April 2026.","pairwise_comparison",["a_gdp","p_gdp"]),
("cmp_dev_010","Compare CPI inflation projections cited in April 2025 and April 2026.","pairwise_comparison",["a_cpi","p_cpi"]),
("cmp_dev_011","Compare crude-oil assumptions or risks in April 2025 and April 2026.","pairwise_comparison",["a_crude","p_crude"]),
("cmp_dev_012","Compare central-government borrowing context in April 2025 and April 2026.","pairwise_comparison",["a_borrow","p_borrow"]),
("trend_dev_001","How did food-inflation conditions evolve across all three reports?","trend_all_reports",["a_food","o_food","p_food"]),
("trend_dev_002","Trace the policy-rate or policy-action evidence across all three reports.","trend_all_reports",["a_repo","o_repo","p_repo"]),
("trend_dev_003","How did crude-oil assumptions or risks evolve across the three MPRs?","trend_all_reports",["a_crude","o_crude","p_crude"]),
("trend_dev_004","Compare global-growth assumptions or language across all reports.","trend_all_reports",["a_global","o_global","p_global"]),
("trend_dev_005","How did central-government borrowing context change across the reports?","trend_all_reports",["a_borrow","o_borrow","p_borrow"]),
("trend_dev_006","How did the neutral policy stance appear across the three reports?","trend_all_reports",["a_stance","o_stance","p_stance"]),
]

unsupported_dev=[
{"question_id":f"unsupported_dev_{i:03d}","question":q,"query_type":"unsupported_period","required_report_ids":[],"ground_truth":{},"expected_answer":"The requested period or premise is unavailable in the registered corpus.","category":"unsupported","source_information_type":[],"split":"dev","verification_status":"verified","verified_against":[],"notes":"Verified against the closed three-report registry."}
for i,q in enumerate(["What did the April 2024 MPR project?","Compare October 2024 with April 2025.","What was RBI's forecast in October 2026?","Did the April 2025 MPR report an April 2027 policy decision?"],1)]

test_specs=[
("single_test_001","What 2025-26 real GDP growth projection was discussed in April 2025?","single_report",["a_gdp"]),
("single_test_002","What 2025-26 CPI inflation projection was stated in April 2025?","single_report",["a_cpi"]),
("single_test_003","What neutral-stance decision was recorded in October 2025?","single_report",["o_stance"]),
("single_test_004","What food-price development was highlighted in October 2025?","single_report",["o_food"]),
("single_test_005","What real GDP growth estimate appears in April 2026?","single_report",["p_gdp"]),
("single_test_006","What revised CPI inflation figure appears in April 2026?","single_report",["p_cpi"]),
("pair_test_001","Compare borrowing evidence in April and October 2025.","pairwise_comparison",["a_borrow","o_borrow"]),
("pair_test_002","Compare neutral-stance evidence in April and October 2025.","pairwise_comparison",["a_stance","o_stance"]),
("pair_test_003","Compare crude-oil evidence in October 2025 and April 2026.","pairwise_comparison",["o_crude","p_crude"]),
("pair_test_004","Compare core-inflation evidence in October 2025 and April 2026.","pairwise_comparison",["o_core","p_core"]),
("pair_test_005","Compare global-growth evidence in April 2025 and April 2026.","pairwise_comparison",["a_global","p_global"]),
("trend_test_001","Trace neutral-stance evidence across all three reports.","trend_all_reports",["a_stance","o_stance","p_stance"]),
("trend_test_002","Trace global-growth evidence across all three reports.","trend_all_reports",["a_global","o_global","p_global"]),
]
unsupported_test=[
{"question_id":"unsupported_test_001","question":"What did the April 2024 MPR say about inflation?","query_type":"unsupported_period","required_report_ids":[],"ground_truth":{},"expected_answer":"April 2024 is not in the registered corpus.","category":"unsupported","source_information_type":[],"split":"test","verification_status":"verified","verified_against":[],"notes":"Verified against registry."},
{"question_id":"unsupported_test_002","question":"Compare October 2025 with October 2026.","query_type":"unsupported_period","required_report_ids":["rbi_mpr_2025_10"],"ground_truth":{},"expected_answer":"October 2026 is not in the registered corpus.","category":"unsupported","source_information_type":[],"split":"test","verification_status":"verified","verified_against":[],"notes":"Verified against registry."},
]

root=Path("data/evaluation")
existing=[json.loads(line) for line in (root/"multi_report_dev.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
existing=[row for row in existing if row["question_id"].startswith("mr_dev_")]
dev=existing+[case(*spec,"dev") for spec in dev_specs]+unsupported_dev
test=[case(*spec,"test") for spec in test_specs]+unsupported_test

def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf-8")
write_jsonl(root/"multi_report_dev.jsonl",dev); write_jsonl(root/"multi_report_test.jsonl",test)

def digest(path): return sha256(path.read_bytes()).hexdigest()
def counts(rows,key): return dict(sorted(Counter(row[key] for row in rows).items()))
manifest={"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),
 "files":{"dev":{"path":"data/evaluation/multi_report_dev.jsonl","sha256":digest(root/"multi_report_dev.jsonl"),"case_count":len(dev)},
          "test":{"path":"data/evaluation/multi_report_test.jsonl","sha256":digest(root/"multi_report_test.jsonl"),"case_count":len(test)}},
 "split_counts":{"dev":len(dev),"test":len(test)},
 "verified_scored_counts":{"dev":sum(r.get("verification_status")=="verified" for r in dev),"test":sum(r.get("verification_status")=="verified" for r in test)},
 "query_type_counts":{"dev":counts(dev,"query_type"),"test":counts(test,"query_type")},
 "category_counts":{"dev":counts(dev,"category"),"test":counts(test,"category")},
 "report_checksums":{rid:sha256(path.read_bytes()).hexdigest() for rid,path in FILES.items()},
 "verification_policy":"Every factual excerpt is normalized and matched exactly against the cited local PDF page before dataset emission.",
 "test_freeze_policy":"Held-out cases are not used for tuning, prompt development, or retrieval debugging.",
 "known_limitations":["PyPDFLoader flattens tables and charts.","The nine inherited April cases use the earlier schema and manually verified pages."]}
(root/"temporal_dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"dev":len(dev),"test":len(test),"dev_types":counts(dev,"query_type"),"test_types":counts(test,"query_type")},indent=2))
