from __future__ import annotations
import re

ACRONYMS=('CPI','GDP','GVA','CRR','MSF','LAF')
TEMPORAL_PATTERNS=(r'\bcompare(?:d with)?\b',r'\bversus\b',r'\bvs\.?\b',
 r'\bbetween\s+(?:april|october|apr|oct)\s+20\d{2}\s+and\s+(?:april|october|apr|oct)\s+20\d{2}\b',
 r'\bacross (?:all|the three) reports\b',r'\bsix months later\b',r'\bin the (?:latest|earliest) report\b')

def normalise_retrieval_query(query:str):
    value=query; removed=[]
    for pattern in TEMPORAL_PATTERNS:
        matches=re.findall(pattern,value,flags=re.I); removed.extend(matches)
        value=re.sub(pattern,' ',value,flags=re.I)
    value=re.sub(r'\bhow did\b',' ',value,flags=re.I)
    value=re.sub(r'\b(change|evolve|changed|evolved)\b',' ',value,flags=re.I)
    value=' '.join(value.replace('?',' ').split())
    return {'original_query':query,'normalised_query':value,'removed_phrases':removed,
            'preserved_numbers':re.findall(r'\b\d+(?:\.\d+)?%?|\b\d{4}-\d{2}',value),
            'preserved_acronyms':[a for a in ACRONYMS if re.search(rf'\b{a}\b',value,re.I)]}

TERMINOLOGY={
 'inflation':['headline inflation','CPI inflation','consumer price inflation'],
 'core inflation':['inflation excluding food and fuel','CPI excluding food and fuel'],
 'growth':['real GDP growth','GDP growth','economic growth','GVA growth'],
 'policy rate':['repo rate','policy repo rate'],
 'liquidity':['system liquidity','banking system liquidity','liquidity adjustment facility','LAF'],
 'food inflation':['food and beverages inflation','food price pressures'],
 'external risks':['external headwinds','global uncertainty','geopolitical risks']}

def expand_query(query:str):
    lower=query.lower(); terms=[]
    for canonical,values in TERMINOLOGY.items():
        if canonical in lower: terms.extend(values[:2])
    return {'original':query,'expanded_query':' '.join([query]+terms),'expansion_terms':terms}

def decompose_facets(query:str,max_facets=4):
    clean=normalise_retrieval_query(query)['normalised_query']
    parts=re.split(r'\s*,\s*|\s+and\s+',clean,flags=re.I)
    facets=[p.strip() for p in parts if p.strip()]
    return facets[:max_facets] if len(facets)>1 else []
