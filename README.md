# Women's Rugby Match Prediction
Predicts match attendance for women's rugby internationals (Six Nations, WXV, Rugby World Cup) using match metadata: competition, round, host nation involvement, date/time trend, format (15s vs 7s).

## How it works
Scrape — pull match-by-match data (date, teams, score, venue, attendance) directly from Wikipedia's {{rugbybox}} / {{#invoke:rugby box}} templates via the MediaWiki API, not by scraping rendered HTML.
Enrich — join in venue capacity from a hand-sourced lookup table, since stadium size is the single biggest confound on raw attendance.
Model — Ridge regression + Random Forest, evaluated with Leave-One-Out cross-validation (appropriate given the dataset is only in the dozens-to-low-hundreds of rows).

## Model Training
1. MAE and Median AE — mean/median absolute error in attendees
2. MAPE and Median APE — as percentages (MAPE is included for familiarity but is disproportionately skewed by small-crowd matches; prefer the median-based metrics for a fair read of accuracy across a dataset spanning ~1,000–80,000+ attendance)
3. Ridge coefficients — for interpretation of which factors actually drive attendance, which is arguably more useful than raw predictive accuracy at this data size

## Current data

As of the latest scrape: 82 rows spanning 2023–2025 Six Nations, 2023–2024 WXV, and the 2025 Rugby World Cup pool stage plus several historical finals. This is enough to get a directionally sensible model (Ridge and Random Forest broadly agree, ~4,700–5,000 MAE), but still small enough that:

- Individual data-quality issues (a bad date, an unmapped team code, a missing venue capacity) can meaningfully move the results — always sanity-check new scrapes before trusting a training run
- Roughly half of all venues seen in the data are still missing a capacity figure at any given time; check the "unmatched venues" output after each scrape
- A handful of VERIFY-flagged capacities are approximate and could be off by a meaningful margin for smaller, less-documented stadiums

## Known limitations
- No team-identity feature. By design, to avoid overfitting a small dataset with 15+ per-team dummy variables. This means the model can't currently distinguish, say, a New Zealand vs. Ireland pool match from a Japan vs. Spain pool match played the same day at the same venue — predictions are driven by context (competition, round, venue, host nation), not team strength. A World Rugby ranking feature is a natural next step once the dataset is larger.
- Some venue capacities are approximate. See VERIFY flags in venue_capacities.csv.
- Attendance is not demand. A sold-out small stadium and an undersold large stadium can report similar attendance figures for very different underlying reasons; venue capacity helps but doesn't fully resolve this.
- MAPE is a misleading headline metric given the wide range of crowd sizes in this dataset (friendlies at ~1,000 up to World Cup finals at ~80,000+) — see the Median APE figure instead.
