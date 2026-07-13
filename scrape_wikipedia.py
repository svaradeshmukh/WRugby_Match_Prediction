"""
Scrape women's rugby match attendance data from Wikipedia by parsing the
raw wikitext {{rugbybox}} templates directly via the MediaWiki API.

This is more reliable than scraping rendered HTML: Wikipedia rugby match
reports use a structured template --

    {{rugbybox
    |id = Ireland v France
    |date = 22 March 2025
    |team1 = {{ruw-rt|IRE}}
    |score = 15-27
    |team2 = {{ruw|FRA}}
    |attendance = 6,976
    |stadium = [[Ravenhill Stadium]], [[Belfast]]
    |referee = ...
    }}

-- with named parameters, which mwparserfromhell can pull out directly
instead of guessing at HTML table structure.

Run on your own machine (needs outbound access to en.wikipedia.org):

    pip install requests mwparserfromhell
    python scrape_wikipedia.py

Edit PAGES_TO_SCRAPE below, or pass Wikipedia page titles on the command
line:

    python scrape_wikipedia.py "2025 Women's Six Nations Championship"
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mwparserfromhell
import pandas as pd
import requests

SCRIPT_VERSION = "2024-07-10-v4-case-insensitive-params"

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "WomensRugbyResearchBot/1.0 (personal research project; contact: set-your-email@example.com)"
}

OUTPUT_CSV = Path("/Users/svaradeshmukh/WRugby_Match_Prediction/data/matches_verified.csv")

PAGES_TO_SCRAPE = [
    "2025 Women's Six Nations Championship",
    "2024 Women's Six Nations Championship",
    "2023 Women's Six Nations Championship",
    "2023 WXV",
    "2024 WXV",
    "2025 Women's Rugby World Cup Pool A",
    "2025 Women's Rugby World Cup Pool B",
    "2025 Women's Rugby World Cup Pool C",
    "2025 Women's Rugby World Cup Pool D",
]

# Rugby country/team template codes -> readable names.
# Extend this as you hit codes not covered here -- the parser will keep
# the raw code (e.g. "RSA") rather than silently guessing if it's missing.
TEAM_CODE_TO_NAME = {
    "IRE": "Ireland", "FRA": "France", "SCO": "Scotland", "WAL": "Wales",
    "ENG": "England", "ITA": "Italy",
    "NZL": "New Zealand", "AUS": "Australia", "CAN": "Canada", "USA": "United States",
    "JPN": "Japan", "RSA": "South Africa", "ZAF": "South Africa", "ESP": "Spain",
    "GER": "Germany", "NED": "Netherlands", "POR": "Portugal", "FIJI": "Fiji", "FIJ": "Fiji",
    "SAM": "Samoa", "TON": "Tonga", "ARG": "Argentina", "BRA": "Brazil",
    "MEX": "Mexico", "KOR": "South Korea", "HKG": "Hong Kong", "SWE": "Sweden",
}


def get_param_ci(template, *names: str) -> str | None:
    """Look up a template parameter by name, tolerant of case and stray
    whitespace (Wikipedia isn't perfectly consistent about this across
    pages/years). Returns the first match among the given candidate names,
    or None if none are present."""
    wanted = {n.strip().lower() for n in names}
    for p in template.params:
        if str(p.name).strip().lower() in wanted:
            return str(p.value)
    return None


@dataclass
class MatchRow:
    match_id: str
    date: str
    competition: str
    round: str
    format: str
    home_team: str
    away_team: str
    home_score: str
    away_score: str
    venue: str
    city: str
    attendance: int
    is_final: int
    is_opener: int
    years_since_prev_wc: str
    source: str


def fetch_wikitext(page_title: str) -> str:
    """Fetch the raw wikitext of a page via the MediaWiki API."""
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": page_title,
        "rvslots": "main",
        "rvprop": "content",
        "format": "json",
        "formatversion": "2",
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0]:
        raise ValueError(f"Page not found: {page_title}")

    return pages[0]["revisions"][0]["slots"]["main"]["content"]


def clean_wikitext(value: str) -> str:
    """Strip wikilinks/templates/refs down to plain readable text."""
    if value is None:
        return ""
    wikicode = mwparserfromhell.parse(str(value))
    text = wikicode.strip_code()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_team_code(team_param: str) -> str:
    """Pull a country/team code out of a team1/team2 param like
    '{{ruw-rt|IRE}}' or '(1 BP) {{ruw-rt|ENG}}', mapping to a readable name."""
    wikicode = mwparserfromhell.parse(team_param)
    templates = wikicode.filter_templates()
    if templates:
        tmpl = templates[0]
        if tmpl.params:
            code = str(tmpl.params[0].value).strip()
            return TEAM_CODE_TO_NAME.get(code, code)
    return clean_wikitext(team_param)


def parse_score(score_param: str) -> tuple[str, str]:
    """Split a score field like '15-27' or '24-21' into (home, away)."""
    cleaned = clean_wikitext(score_param)
    parts = re.split(r"[\u2013\u2012\-]", cleaned)
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return "", ""


def parse_stadium(stadium_param: str) -> tuple[str, str]:
    """Split '[[Ravenhill Stadium]], [[Belfast]]' into (venue, city)."""
    cleaned = clean_wikitext(stadium_param)
    parts = [p.strip() for p in cleaned.split(",")]
    venue = parts[0] if parts else ""
    city = parts[1] if len(parts) > 1 else ""
    return venue, city


def parse_date(date_param: str) -> str:
    cleaned = clean_wikitext(date_param)
    try:
        return pd.to_datetime(cleaned).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return cleaned


def slugify(date: str, home: str, away: str) -> str:
    base = f"{date}_{home}_{away}".lower()
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_")


def guess_competition(page_title: str) -> str:
    if "Six Nations" in page_title:
        return "Six Nations"
    if "WXV" in page_title:
        return "WXV"
    if "Rugby World Cup" in page_title:
        return "Rugby World Cup"
    return "Unknown"


def is_rugbybox_template(template_name: str) -> bool:
    """Wikipedia uses at least two different invocations for the same
    match-report content: the {{rugbybox}} template (Six Nations pages)
    and the {{#invoke:rugby box|main}} Lua module call (World Cup pool
    pages). Both carry the same named parameters underneath."""
    name = template_name.strip().lower()
    return name == "rugbybox" or name.startswith("#invoke:rugby box")


def find_all_headers(wikitext: str) -> list[tuple[int, str]]:
    """Every '== Heading ==' / '=== Heading ===' regardless of level,
    in document order with character position."""
    pattern = re.compile(r"^=+\s*(.+?)\s*=+\s*$", re.MULTILINE)
    return [(m.start(), m.group(1).strip()) for m in pattern.finditer(wikitext)]


ROUND_KEYWORDS = re.compile(r"\b(pool|round|matchday|semi.?final|quarter.?final|final)\b", re.IGNORECASE)
ROUND_KEYWORDS_WITH_LETTER = re.compile(r"\b(Pool [A-Z]|Round \d+|Semi.?final|Quarter.?final|Final)\b", re.IGNORECASE)


def round_label_for_position(pos: int, headers: list[tuple[int, str]]) -> str:
    """Walk backward through ALL headers (any level) and return the most
    recent one that actually looks like a round/pool/stage label, skipping
    per-match headers like 'England vs United States' along the way."""
    current = ""
    for header_pos, header_text in headers:
        if header_pos > pos:
            break
        if ROUND_KEYWORDS.search(header_text):
            current = header_text
    return current


def nearest_header(pos: int, headers: list[tuple[int, str]]) -> str:
    """The single closest preceding header regardless of content -- on pool
    pages this is usually the '=== Team A vs Team B ===' match subheading,
    useful as a fallback for team names / final-detection."""
    current = ""
    for header_pos, header_text in headers:
        if header_pos <= pos:
            current = header_text
        else:
            break
    return current


def parse_rugbyboxes(wikitext: str, page_title: str) -> list[MatchRow]:
    rows: list[MatchRow] = []
    competition = guess_competition(page_title)
    headers = find_all_headers(wikitext)

    wikicode = mwparserfromhell.parse(wikitext)
    for template in wikicode.filter_templates(matches=lambda t: is_rugbybox_template(t.name)):
        try:
            attendance_raw = get_param_ci(template, "attendance") or ""
            attendance_match = re.search(r"[\d,]+", attendance_raw)
            if not attendance_match:
                continue  # no attendance data -- skip, don't guess

            attendance = int(attendance_match.group().replace(",", ""))

            date_raw = get_param_ci(template, "date")
            date = parse_date(date_raw) if date_raw else ""

            # Different pages use team1/team2 in some years, home/away in others
            home_raw = get_param_ci(template, "team1", "home")
            away_raw = get_param_ci(template, "team2", "away")
            home_team = extract_team_code(home_raw) if home_raw else ""
            away_team = extract_team_code(away_raw) if away_raw else ""

            score_raw = get_param_ci(template, "score")
            home_score, away_score = parse_score(score_raw) if score_raw else ("", "")

            stadium_raw = get_param_ci(template, "stadium", "venue")
            venue, city = parse_stadium(stadium_raw) if stadium_raw else ("", "")

            template_str = str(template)
            pos = wikitext.find(template_str)

            # 'id' param (Six Nations pages) vs subsection header fallback (World Cup pool pages)
            id_raw = get_param_ci(template, "id")
            match_label = clean_wikitext(id_raw) if id_raw else ""
            if not match_label and pos != -1:
                match_label = nearest_header(pos, headers)

            # If team1/team2/home/away were missing or unresolved, fall back to
            # splitting the "Team A v Team B" / "Team A vs Team B" subsection header/id.
            if (not home_team or not away_team) and match_label:
                vs_match = re.split(r"\s+v(?:s\.?)?\s+", match_label, flags=re.IGNORECASE)
                if len(vs_match) == 2:
                    home_team = home_team or vs_match[0].strip()
                    away_team = away_team or vs_match[1].strip()

            home_team = home_team or "Unknown"
            away_team = away_team or "Unknown"

            round_label = round_label_for_position(pos, headers) if pos != -1 else ""
            if not round_label:
                title_round_match = ROUND_KEYWORDS_WITH_LETTER.search(page_title)
                if title_round_match:
                    round_label = title_round_match.group(0)
            is_final = int("final" in round_label.lower() or "final" in match_label.lower())

            rows.append(
                MatchRow(
                    match_id=slugify(date, home_team, away_team),
                    date=date,
                    competition=competition,
                    round=round_label,
                    format="15s",
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home_score,
                    away_score=away_score,
                    venue=venue,
                    city=city,
                    attendance=attendance,
                    is_final=is_final,
                    is_opener=0,
                    years_since_prev_wc="",
                    source=f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
                )
            )
        except (AttributeError, ValueError) as e:
            print(f"  Skipping a malformed rugbybox template: {e}")
            continue

    return rows


def scrape_page(page_title: str) -> list[MatchRow]:
    print(f"Fetching '{page_title}' ...")
    wikitext = fetch_wikitext(page_title)
    rows = parse_rugbyboxes(wikitext, page_title)

    if not rows:
        print(f"  WARNING: no rugbybox templates with attendance found -- check manually.")
    else:
        print(f"  Found {len(rows)} matches with attendance data.")

    return rows


def merge_into_csv(new_rows: list[MatchRow]) -> None:
    new_df = pd.DataFrame([asdict(r) for r in new_rows])

    if OUTPUT_CSV.exists():
        existing_df = pd.read_csv(OUTPUT_CSV)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["match_id"], keep="first")
    else:
        combined = new_df

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(combined)} total rows to {OUTPUT_CSV}")


def main():
    print(f"scrape_wikipedia.py version: {SCRIPT_VERSION}\n")
    pages = sys.argv[1:] if len(sys.argv) > 1 else PAGES_TO_SCRAPE

    all_rows: list[MatchRow] = []
    for title in pages:
        try:
            all_rows.extend(scrape_page(title))
        except (requests.RequestException, ValueError) as e:
            print(f"  ERROR fetching '{title}': {e}")
        time.sleep(1)  # be polite to Wikipedia's servers

    if not all_rows:
        print("\nNo rows scraped. Check page titles are exact (including apostrophes)")
        print("and that these pages still use the {{rugbybox}} template.")
        return

    print(f"\nTotal scraped rows before dedup/merge: {len(all_rows)}")
    preview = pd.DataFrame([asdict(r) for r in all_rows[:10]])
    print(preview.to_string(index=False))

    merge_into_csv(all_rows)


if __name__ == "__main__":
    main()
