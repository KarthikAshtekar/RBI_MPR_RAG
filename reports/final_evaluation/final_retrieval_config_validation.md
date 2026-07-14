# Final Retrieval Configuration Validation

Status: passed
Config: `configs\final_retrieval_selected.yaml`
Selected experiment: `ADJ00`
File SHA-256: `cee93262310a59be1beb6ba62ce7ceb8c65d42ecc7e53604a77a06951f73ca33`
Configuration checksum: `4b9e74513eb0f16ec9d13c7158a848aa75f245764d86906b5e545376a001f945`

| Field | Expected | Actual | Match |
|---|---:|---:|---:|
| `id` | `ADJ00` | `ADJ00` | True |
| `retrieval.dense_k` | `50` | `50` | True |
| `retrieval.bm25_k` | `50` | `50` | True |
| `fusion.retain` | `30` | `30` | True |
| `fusion.k` | `60` | `60` | True |
| `context_selection.single` | `6` | `6` | True |
| `context_selection.pairwise` | `5` | `5` | True |
| `context_selection.trend` | `4` | `4` | True |
| `context_selection.structural_mode` | `none` | `none` | True |
| `chunking.strategy` | `recursive` | `recursive` | True |
| `chunking.chunk_size` | `1000` | `1000` | True |
| `chunking.chunk_overlap` | `300` | `300` | True |
| `embedding.model` | `sentence-transformers/all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` | True |
| `reranker.model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | True |
| `configuration_checksum` | `4b9e74513eb0f16ec9d13c7158a848aa75f245764d86906b5e545376a001f945` | `4b9e74513eb0f16ec9d13c7158a848aa75f245764d86906b5e545376a001f945` | True |
| `evaluation.heldout_loaded` | `False` | `False` | True |
