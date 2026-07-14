# MMR Results for Presentation

MMR was tested to see whether a diversity-aware context-selection step could reduce repeated context while preserving evidence completeness.

Formula: `lambda * relevance_score - (1 - lambda) * max_similarity_to_selected_documents`.

MRR is Mean Reciprocal Rank, a metric. MMR is Maximal Marginal Relevance, a selection technique.

- MMR_BASELINE_V2_COHERE: CER=0.4666666666666667, Hit=0.5, Evidence=0.6, Macro MRR=0.41537037037037033, Repeated=0.0
- MMR_LAMBDA_06: CER=0.5333333333333333, Hit=0.5666666666666667, Evidence=0.65, Macro MRR=0.4054629629629629, Repeated=0.0
- MMR_LAMBDA_07: CER=0.5, Hit=0.5333333333333333, Evidence=0.6333333333333333, Macro MRR=0.41453703703703704, Repeated=0.0
- MMR_LAMBDA_08: CER=0.4666666666666667, Hit=0.5, Evidence=0.6, Macro MRR=0.4163888888888889, Repeated=0.0

MMR improved Hit Rate: True
MMR improved Macro MRR: False
MMR was selected: True

Interview-ready explanation: MMR was evaluated as a controlled diversity-selection layer after Cohere reranking. It was only eligible if evidence completeness stayed intact; diversity alone was not enough to replace the selected V2 Cohere retrieval system.
