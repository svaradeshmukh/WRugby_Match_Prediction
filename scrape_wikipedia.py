"""
Scrape women's rugby match attendance data from Wikipedia.

IMPORTANT: run this on your own machine, not in a sandboxed environment
with restricted network access -- it needs to reach en.wikipedia.org.

    pip install requests beautifulsoup4 pandas lxml
    python scrape_wikipedia.py

Wikipedia rugby articles use two different table layouts depending on
the page, so this script tries both:

  1. "Match report" boxes (class="vevent" or similar match-summary
     templates) -- used on individual World Cup match pages and some
     Six Nations round pages. Structured infobox-style: date, venue,
     attendance, referee each on their own line.

  2. Plain results tables -- used on some Six Nations season pages and
     WXV pages. A single <table class="wikitable"> with columns like
     Date | Home | Score | Away | Venue | Attendance.

The script tries (1) first, falls back to (2), and if neither finds an
"Attendance" field for a given page, it skips that page and logs it so
you can check it manually -- it does NOT silently invent a number.

Usage
-----
Edit PAGES_TO_SCRAPE below with the Wikipedia URLs you want, or pass
them on the command line:

    python scrape_wikipedia.py "https://en.wikipedia.org/wiki/2025_Women%27s_Six_Nations_Championship"
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, asdict
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    # Wikipedia asks bots/scripts to identify themselves - be a good citizen
    "User-Agent": "WomensRugbyResearchBot/1.0 (personal research project; contact: set-your-email@example.com)"
}

OUTPUT_CSV = Path(__file__).parent.parent / "data" / "matches_verified.csv"

PAGES_TO_SCRAPE = [
    "https://en.wikipedia.org/wiki/2025_Women%27s_Six_Nations_Championship",
    "https://en.wikipedia.org/wiki/2024_Women%27s_Six_Nations_Championship",
    "https://en.wikipedia.org/wiki/2023_Women%27s_Six_Nations_Championship",
    "https://en.wikipedia.org/wiki/2025_WXV",
    "https://en.wikipedia.org/wiki/2024_WXV",
    "https://en.wikipedia.org/wiki/2025_Women%27s_Rugby_World_Cup_Pool_A",
    "https://en.wikipedia.org/wiki/2025_Women%27s_Rugby_World_Cup_Pool_B",
    "https://en.wikipedia.org/wiki/2025_Women%27s_Rugby_World_Cup_Pool_C",
    "https://en.wikipedia.org/wiki/2025_Women%27s_Rugby_World_Cup_Pool_D",
]


@dataclass
class MatchRow:
    match_id: str
    date: str
    competition: str
    round: str
    format: str
    home_team: str
    away_team: str
    venue: str
    city: str
    attendance: int
    is_final: int
    is_opener: int
    years_since_prev_wc: str
    source: str


def fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def slugify(date: str, home: str, away: str) -> str:
    base = f"{date}_{home}_{away}".lower()
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_")


def guess_competition(url: str) -> str:
    if "Six_Nations" in url:
        return "Six Nations"
    if "WXV" in url:
        return "WXV"
    if "Rugby_World_Cup" in url:
        return "Rugby World Cup"
    return "Unknown"


def parse_matchboxes(soup: BeautifulSoup, source_url: str) -> list[MatchRow]:
    """Strategy 1: individual match-report infobox tables (class 'vevent' or similar)."""
    rows: list[MatchRow] = []
    competition = guess_competition(source_url)

    # Wikipedia rugby match reports commonly use tables with class containing
    # 'football' (a reused template family) or an explicit 'vevent' microformat.
    boxes = soup.select("table.vevent, table.football-match-report, table.footballbox")

    for box in boxes:
        text = box.get_text(" ", strip=True)

        attendance_match = re.search(r"Attendance[:\s]*([\d,]+)", text)
        if not attendance_match:
            continue  # no attendance data in this box -- skip, don't guess
        attendance = int(attendance_match.group(1).replace(",", ""))

        venue_match = re.search(r"Venue[:\s]*([^,\n]+?)(?=\s*(?:Attendance|Referee|$))", text)
        venue = venue_match.group(1).strip() if venue_match else ""

        date_match = re.search(r"(\d{1,2} \w+ \d{4})", text)
        date = date_match.group(1) if date_match else ""
        if date:
            try:
                date = pd.to_datetime(date).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        # Team names: typically in <th> or specific score-cell classes
        team_cells = box.select(".fhome, .fteam1, th.team-a") or box.select("th")
        home_team = team_cells[0].get_text(strip=True) if team_cells else "Unknown"
        away_cells = box.select(".faway, .fteam2, th.team-b")
        away_team = away_cells[0].get_text(strip=True) if away_cells else "Unknown"

        rows.append(
            MatchRow(
                match_id=slugify(date, home_team, away_team),
                date=date,
                competition=competition,
                round="",  # fill in manually -- not reliably parseable from the box alone
                format="15s",
                home_team=home_team,
                away_team=away_team,
                venue=venue,
                city="",
                attendance=attendance,
                is_final=0,
                is_opener=0,
                years_since_prev_wc="",
                source=source_url,
            )
        )

    return rows


def parse_results_table(soup: BeautifulSoup, source_url: str) -> list[MatchRow]:
    """Strategy 2: plain wikitable with an 'Attendance' column."""
    rows: list[MatchRow] = []
    competition = guess_competition(source_url)

    try:
        tables = pd.read_html(StringIO(str(soup)))
    except ValueError:
        return rows

    for table in tables:
        cols_lower = [str(c).lower() for c in table.columns]
        if not any("attendance" in c for c in cols_lower):
            continue

        col_map = {c.lower(): c for c in table.columns}
        attendance_col = next(c for c in table.columns if "attendance" in str(c).lower())

        date_col = col_map.get("date")
        home_col = col_map.get("home") or col_map.get("home team")
        away_col = col_map.get("away") or col_map.get("away team")
        venue_col = col_map.get("venue") or col_map.get("stadium")

        for _, r in table.iterrows():
            raw_att = str(r.get(attendance_col, "")).replace(",", "")
            att_match = re.search(r"\d+", raw_att)
            if not att_match:
                continue
            attendance = int(att_match.group())

            date_val = str(r.get(date_col, "")) if date_col else ""
            try:
                date_val = pd.to_datetime(date_val).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

            home = str(r.get(home_col, "Unknown")) if home_col else "Unknown"
            away = str(r.get(away_col, "Unknown")) if away_col else "Unknown"
            venue = str(r.get(venue_col, "")) if venue_col else ""

            rows.append(
                MatchRow(
                    match_id=slugify(date_val, home, away),
                    date=date_val,
                    competition=competition,
                    round="",
                    format="15s",
                    home_team=home,
                    away_team=away,
                    venue=venue,
                    city="",
                    attendance=attendance,
                    is_final=0,
                    is_opener=0,
                    years_since_prev_wc="",
                    source=source_url,
                )
            )

    return rows


def scrape_page(url: str) -> list[MatchRow]:
    print(f"Fetching {url} ...")
    soup = fetch(url)

    rows = parse_matchboxes(soup, url)
    if not rows:
        rows = parse_results_table(soup, url)

    if not rows:
        print(f"  WARNING: no attendance data found on this page -- check manually: {url}")
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
    pages = sys.argv[1:] if len(sys.argv) > 1 else PAGES_TO_SCRAPE

    all_rows: list[MatchRow] = []
    for url in pages:
        try:
            all_rows.extend(scrape_page(url))
        except requests.RequestException as e:
            print(f"  ERROR fetching {url}: {e}")
        time.sleep(1)  # be polite to Wikipedia's servers

    if not all_rows:
        print("\nNo rows scraped. Wikipedia's table structure may not match this")
        print("script's assumptions for these pages -- open one in a browser,")
        print("inspect the table HTML, and adjust parse_matchboxes/parse_results_table.")
        return

    print(f"\nTotal scraped rows before dedup/merge: {len(all_rows)}")
    print("\nReview these before merging -- team names, venues, and rounds")
    print("often need manual cleanup after scraping (Wikipedia markup varies")
    print("page to page). Printing first 10 for a sanity check:\n")
    preview = pd.DataFrame([asdict(r) for r in all_rows[:10]])
    print(preview.to_string(index=False))

    merge_into_csv(all_rows)


if __name__ == "__main__":
    main()
