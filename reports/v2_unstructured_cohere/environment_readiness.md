# V2 Environment Readiness

COHERE_API_KEY available: True
UNSTRUCTURED_API_KEY available: True
GROQ_API_KEY available: True
python-dotenv installed: True
Project .env loaded: True
unstructured installed: True (0.24.1)
cohere installed: True

Install optional V2 dependencies with:

```powershell
python -m pip install -r requirements-v2.txt
```

## V2 resource readiness

- Experiment filter: ['V2_UNSTRUCTURED_COHERE', 'V2_UNSTRUCTURED_ONLY']
- Unstructured resources available: False
- Unstructured block reason: RuntimeError: Unstructured extraction failed for rbi_mpr_2025_04: OCRUnavailable: OCR fallback was skipped because tesseract is not installed or not on PATH.
