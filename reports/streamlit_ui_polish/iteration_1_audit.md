# Iteration 1 Screenshot Audit

Screenshots were captured successfully and no Streamlit crash/error text was visible.

## Findings

- Header, caveat, and top metrics were clear and readable on desktop.
- The saved-answer/citation section did not appear in the visible answer/citation screenshots because it was below the fold and the scroll automation did not target Streamlit’s scroll container correctly.
- Sidebar was functional but long model/prompt identifiers wrapped awkwardly.
- Main issue to fix: make the answer/citation flow visible immediately after the user sees the query controls.

## Fix plan

- Move the answer/citation panel beside the query controls.
- Shorten sidebar model/prompt display.
- Update screenshot script to use wheel scrolling for Streamlit pages.
