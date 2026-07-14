# Current Streamlit App Audit

Audit source: `streamlit_app.py` before UI polish.

## 1. Page structure

- Uses a title, caption, sidebar, and three tabs: Ask / Demo, Method comparison, Limitations.
- Functional but tab-first structure hides the core demo and final metrics from the first visual screen.

## 2. Sidebar clutter

- Sidebar is not excessive, but pipeline text is dense and not visually prioritized.
- No status badge or concise selected-system summary.

## 3. Metric display

- Retrieval metrics are shown only inside the Method comparison tab.
- Generation metrics are displayed as a full raw dataframe, which is too dense for a demo.
- Final MMR06 selected system is not clearly surfaced; old V2 sufficiency row can appear as best generation.

## 4. Query/demo flow

- Saved-example demo exists and does not require live APIs.
- There is no explicit action button; answer updates immediately from the text area.
- User may not understand whether the answer is saved or live.

## 5. Citation display

- Citations render as a raw dataframe.
- Citations are not grouped by report period.
- Retrieved evidence is available in expanders, but source labels are long and visually heavy.

## 6. Method comparison display

- Includes broad method table, but it is shown as an unstyled dataframe.
- MRR/MMR distinction is not explained near the comparison.

## 7. Limitations display

- Development-only and not-production-ready caveats exist.
- Limitations are isolated in a tab, so they may be missed in screenshots of the main page.

## 8. Mobile/responsive risk

- Wide dataframes and four metric columns may overflow on small/mobile screens.
- Long raw generated answer/citations can create dense vertical blocks.

## 9. Error handling

- Streamlit missing is handled with a console message.
- Missing demo artifacts show a warning.
- Missing comparison artifacts degrade to empty rows, but the UI does not clearly explain fallback status.

## 10. Live API key dependency

- Opening the app does not require Groq/Cohere keys.
- App text mentions live mode, but there is no real live-mode toggle; this should be clarified.

## 11. Saved-example demo mode

- Exists via `reports/v2_sufficiency/dev_generation_sufficiency_raw_results.json`.
- Should be updated to prefer the final selected MMR06 bake-off generation artifacts.

## 12. Missing artifact handling

- Basic JSON fallback exists.
- Needs a clean UI-level fallback message and helper tests.

## Initial UI score estimate

- Visual clarity: 12/20
- Simplicity: 9/15
- User flow: 9/15
- Trust/transparency: 10/15
- Responsiveness: 5/10
- Error handling: 6/10
- Demo readiness: 8/15
- Total: 59/100
