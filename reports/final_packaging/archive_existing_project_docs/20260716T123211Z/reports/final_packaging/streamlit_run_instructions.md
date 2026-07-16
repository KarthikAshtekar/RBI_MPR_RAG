# Streamlit Run Instructions

Status: `available`

The app runs in safe saved-example demo mode using development evaluation artifacts. It does not call live Groq or Cohere APIs and does not require API keys to open.

```powershell
python -m pip install -r requirements-v2.txt
streamlit run streamlit_app.py
```

Optional fixed-port run:

```powershell
streamlit run streamlit_app.py --server.port 8501 --server.headless true
```

UI polish validation artifacts:

- `reports/streamlit_ui_polish/final_streamlit_ui_report.md`
- `reports/streamlit_ui_polish/screenshots/iteration_3/`

The app displays development evaluation results only. It is not production-ready; it is demo/interview-ready with known limitations.
