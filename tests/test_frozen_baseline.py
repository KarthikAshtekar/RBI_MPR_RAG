from hashlib import sha256
from pathlib import Path

EXPECTED={
"baseline_report.md":"311fa2aab13c649c244a5740045859a8acf494db84711b29bc442852a8e6446e",
"retrieval_raw_results.json":"cc3bb9406ac96c19a292a4c04e878dad32c1a641a7da5437145f828d977cb9a1",
"retrieval_question_results.csv":"ea7f47d8bc32276268e87808ba671522b18de5be9c0a0a404ac977ed05b069b7",
"retrieval_pipeline_summary.csv":"3088dfce0f9dfbb4be25406f3315975399be501dd957eb30f1047692cc0d007e",
"retrieval_summary.json":"8a276c1839dd37c690cd35cd6a506824f95bf71023f5bc5714c9b0e899665429"}

def test_frozen_baseline_files_are_byte_identical():
    root=Path("reports/current")
    assert {name:sha256((root/name).read_bytes()).hexdigest() for name in EXPECTED}==EXPECTED
