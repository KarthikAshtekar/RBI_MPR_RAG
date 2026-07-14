# Retrieval-to-Generation Analysis

Rows analysed: 34
Complete retrieval cases: 14
Mean factual score when retrieval complete: 0.6942262207299802
Incomplete retrieval cases: 16
Abstention rate when retrieval incomplete: 0.0
Table/numeric cases: 21
Table/numeric mean factual score: 0.6549121147111432
Macro MRR to factual-score correlation: 0.33853171948154576

## Required diagnostic answers

1. When retrieval is complete, does generation answer correctly? Generation is materially better when retrieval is complete, but not perfect.
2. When retrieval is incomplete, does generation abstain or hallucinate? Incomplete retrieval did not reliably trigger abstention.
3. Are temporal attribution errors caused by retrieval or generation? Temporal attribution errors should be inspected row-by-row using citation and retrieval traces.
4. Are table/numeric questions still weak after Cohere retrieval? Table/numeric questions remain a material weakness area.
5. Does better Macro MRR translate into better answer quality? Macro MRR has a positive but modest diagnostic correlation with factual answer quality.

Interpretation: these are development-only diagnostic links between saved retrieval outcomes and generated answers.
