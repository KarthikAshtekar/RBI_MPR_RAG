from rbi_rag.query_optimisation import normalise_retrieval_query,expand_query,decompose_facets
from rbi_rag.bm25_preprocessing import finance_tokens

def test_query_normalisation_preserves_finance_values_and_acronyms():
    result=normalise_retrieval_query('Compare CPI and GDP at 6.25% and 25 basis points in FY2025-26 versus April 2025')
    assert '6.25%' in result['normalised_query'] and '25' in result['normalised_query']
    assert {'CPI','GDP'} <= set(result['preserved_acronyms'])

def test_terminology_expansion_is_bounded_and_traceable():
    result=expand_query('core inflation and growth')
    assert result['original']=='core inflation and growth' and result['expansion_terms']

def test_facets_are_deterministic_and_bounded():
    assert decompose_facets('growth projections and inflation risks')==decompose_facets('growth projections and inflation risks')
    assert len(decompose_facets('a, b, c, d, e'))<=4

def test_finance_tokenisation_preserves_required_tokens():
    tokens=finance_tokens('6.25% 25 bps FY2025-26 2025-26 CPI GDP GVA CRR MSF LAF repo-rate food-and-fuel')
    for token in ('6.25%','25','bps','FY2025-26','2025-26','CPI','GDP','GVA','CRR','MSF','LAF','repo-rate','food-and-fuel'):
        assert token in tokens
