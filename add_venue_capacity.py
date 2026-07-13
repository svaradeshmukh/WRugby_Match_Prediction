"""
Merge venue_capacities.csv onto matches_verified.csv by venue name.

Wikipedia refers to the same physical stadium differently across pages
and years (naming-rights sponsors change, some pages use the city-park
name, others the sponsor name) -- e.g. "Twickenham Stadium" vs "Allianz
Stadium (Twickenham)" are the same ground. This script normalizes via
an explicit alias map rather than fuzzy string matching, since fuzzy
matching on stadium names is a good way to silently merge the wrong
venue together.

Run from the src/ directory:
    python add_venue_capacity.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MATCHES_PATH = Path("/Users/svaradeshmukh/WRugby_Match_Prediction/data/matches_verified.csv")
CAPACITY_PATH = Path("/Users/svaradeshmukh/WRugby_Match_Prediction/venue_capacities.csv")
OUTPUT_PATH = Path("/Users/svaradeshmukh/WRugby_Match_Prediction/data/matches_with_capacity.csv")

# Map venue names as they appear in matches_verified.csv -> canonical name
# used in venue_capacities.csv. Add to this as new venue-name variants
# show up (a mismatch here shows up as a blank venue_capacity, not a
# wrong one, so it's a safe failure mode -- see the "unmatched" report
# printed at the end of this script).
VENUE_ALIASES = {
    "Allianz Stadium (Twickenham)": "Twickenham Stadium",
    "Twickenham": "Twickenham Stadium",
}


def load_capacity_lookup() -> dict[str, int]:
    df = pd.read_csv(CAPACITY_PATH)
    return dict(zip(df["venue"], df["capacity"]))


def resolve_capacity(venue: str, lookup: dict[str, int]) -> int | None:
    canonical = VENUE_ALIASES.get(venue, venue)
    return lookup.get(canonical)


def main():
    matches = pd.read_csv(MATCHES_PATH)
    lookup = load_capacity_lookup()

    matches["venue_capacity"] = matches["venue"].apply(lambda v: resolve_capacity(v, lookup))

    unmatched = matches[matches["venue_capacity"].isna()]["venue"].unique()
    if len(unmatched):
        print("No capacity found for these venues -- add them to venue_capacities.csv")
        print("or add an alias to VENUE_ALIASES if it's a naming variant of a venue")
        print("already in the lookup:")
        for v in unmatched:
            print(f"  - {v}")
        print()

    matches.to_csv(OUTPUT_PATH, index=False)
    matched_count = matches["venue_capacity"].notna().sum()
    print(f"Wrote {OUTPUT_PATH} -- {matched_count}/{len(matches)} rows matched a venue capacity.")


if __name__ == "__main__":
    main()
