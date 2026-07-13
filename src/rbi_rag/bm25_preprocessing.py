from __future__ import annotations
import re
STOP={'the','a','an','of','and','to','in','was','were','is','are'}
TOKEN=re.compile(r'\b(?:FY)?\d{4}-\d{2}\b|\b\d+(?:\.\d+)?%?|\b[A-Za-z]+(?:-[A-Za-z]+)*\b')
def finance_tokens(text:str,remove_stopwords=False):
    values=TOKEN.findall(text)
    return [v for v in values if not remove_stopwords or v.lower() not in STOP]
