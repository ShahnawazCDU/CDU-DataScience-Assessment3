"""
build_datasets.py
HIT140 Assessment 3 - Objective 2 (provenance script; not part of the notebook analysis)
Rebuilds the two modelling datasets from the raw sources:
  * reg21_dataset.csv  (104 rows, 1 per match)      -> Linear Regression 2.1
  * reg22_dataset.csv  (208 rows, 1 per team-match) -> Linear Regression 2.2
Usage:  python src/build_datasets.py [output_dir]      (default: data/rebuilt)
The submitted datasets in data/ were produced by this script; the notebook re-runs it into
data/rebuilt/ and checks the result is identical, without touching the originals.

Raw sources (data/raw):
  * worldcup2026.json          - openfootball/worldcup.json (all 104 WC2026 fixtures and scores)
  * international_results.csv  - martj42/international_results (every men's international since 1872)
  * shootouts.csv              - martj42/international_results (penalty shoot-out winners)

EVERY explanatory variable is computed only from information dated strictly BEFORE
the kick-off of the match it describes (the "pre-match rule" in the brief).
Scores used as targets include extra time but exclude penalty shoot-outs.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "rebuilt"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load and harmonise the two sources
# ---------------------------------------------------------------------------
NAME_MAP = {"USA": "United States", "Bosnia & Herzegovina": "Bosnia and Herzegovina"}
HOSTS = {"United States", "Mexico", "Canada"}

# Stadium coordinates (lat, lon) for the 16 host venues, used for travel distance
VENUES = {
    "Mexico City": (19.3029, -99.1505), "Guadalajara (Zapopan)": (20.6817, -103.4627),
    "Monterrey (Guadalupe)": (25.6694, -100.2446), "Toronto": (43.6332, -79.4186),
    "Vancouver": (49.2768, -123.1120), "Seattle": (47.5952, -122.3316),
    "San Francisco Bay Area (Santa Clara)": (37.4030, -121.9700),
    "Los Angeles (Inglewood)": (33.9535, -118.3392), "Dallas (Arlington)": (32.7473, -97.0945),
    "Houston": (29.6847, -95.4107), "Kansas City": (39.0489, -94.4839),
    "Atlanta": (33.7554, -84.4008), "Miami (Miami Gardens)": (25.9580, -80.2389),
    "Boston (Foxborough)": (42.0909, -71.2643), "Philadelphia": (39.9008, -75.1675),
    "New York/New Jersey (East Rutherford)": (40.8135, -74.0745),
}
VENUE_COUNTRY = {"Mexico City": "Mexico", "Guadalajara (Zapopan)": "Mexico",
                 "Monterrey (Guadalupe)": "Mexico", "Toronto": "Canada", "Vancouver": "Canada"}


def haversine_km(a, b):
    (la1, lo1), (la2, lo2) = map(lambda p: (math.radians(p[0]), math.radians(p[1])), (a, b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


wc_json = json.load(open(RAW / "worldcup2026.json", encoding="utf-8"))["matches"]
rows = []
for i, m in enumerate(wc_json, start=1):
    sc = m["score"]
    final = sc.get("et") or sc["ft"]          # score after extra time if played, no shoot-out
    ko = not m["round"].startswith("Matchday")
    rows.append({
        "match_id": i, "date": pd.Timestamp(m["date"]), "stage": m["round"] if ko else "Group stage",
        "group": m.get("group", ""), "venue": m["ground"],
        "team1": NAME_MAP.get(m["team1"], m["team1"]), "team2": NAME_MAP.get(m["team2"], m["team2"]),
        "goals1": final[0], "goals2": final[1], "extra_time": int("et" in sc), "knockout": int(ko),
    })
wc = pd.DataFrame(rows)
assert len(wc) == 104 and wc.match_id.is_unique

res = pd.read_csv(RAW / "international_results.csv", parse_dates=["date"])
res = res.dropna(subset=["home_score", "away_score"]).sort_values("date", kind="stable").reset_index(drop=True)

# Cross-source validation of all 104 scores (scientific rigour: two independent sources must agree)
wc_res = res[(res.tournament == "FIFA World Cup") & (res.date >= "2026-06-01")]
mismatch = 0
for _, m in wc.iterrows():
    hit = wc_res[(wc_res.date == m.date) & (
        ((wc_res.home_team == m.team1) & (wc_res.away_team == m.team2)) |
        ((wc_res.home_team == m.team2) & (wc_res.away_team == m.team1)))]
    assert len(hit) == 1, f"match {m.match_id} not found in second source"
    h = hit.iloc[0]
    g1, g2 = (h.home_score, h.away_score) if h.home_team == m.team1 else (h.away_score, h.home_score)
    mismatch += int((g1, g2) != (m.goals1, m.goals2))
print(f"Score cross-check: {104 - mismatch}/104 matches agree between the two sources")
assert mismatch == 0

# ---------------------------------------------------------------------------
# 2. World Football Elo ratings, rebuilt match-by-match from 1872
#    (eloratings.net method: K by competition, goal-margin multiplier, +100 home edge)
# ---------------------------------------------------------------------------
def k_factor(t):
    if t == "FIFA World Cup":
        return 60
    if t in {"UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup",
             "Confederations Cup", "CONCACAF Championship", "OFC Nations Cup"}:
        return 50
    if "qualification" in t or "Nations League" in t:
        return 40
    if t == "Friendly":
        return 20
    return 30


def margin_mult(gd):
    gd = abs(gd)
    return 1 if gd <= 1 else 1.5 if gd == 2 else 1.75 + (gd - 3) / 8


elo, pre_elo = {}, {}            # pre_elo[(date, home, away)] = (elo_home, elo_away) BEFORE the match
for r in res.itertuples(index=False):
    eh, ea = elo.get(r.home_team, 1500.0), elo.get(r.away_team, 1500.0)
    pre_elo[(r.date, r.home_team, r.away_team)] = (eh, ea)
    dr = eh - ea + (0 if r.neutral else 100)
    exp_h = 1 / (1 + 10 ** (-dr / 400))
    gd = r.home_score - r.away_score
    w = 1.0 if gd > 0 else 0.0 if gd < 0 else 0.5   # shoot-outs count as draws, as in Elo
    delta = k_factor(r.tournament) * margin_mult(gd) * (w - exp_h)
    elo[r.home_team], elo[r.away_team] = eh + delta, ea - delta


def elo_before(m, team):
    for key in [(m.date, m.team1, m.team2), (m.date, m.team2, m.team1)]:
        if key in pre_elo:
            a, b = pre_elo[key]
            return a if key[1] == team else b
    raise KeyError(m.match_id)

# ---------------------------------------------------------------------------
# 3. Team-match history table (long format) for form, rest, head-to-head, WC experience
# ---------------------------------------------------------------------------
long = pd.concat([
    res.assign(team=res.home_team, opp=res.away_team, gf=res.home_score, ga=res.away_score),
    res.assign(team=res.away_team, opp=res.home_team, gf=res.away_score, ga=res.home_score),
])[["date", "team", "opp", "gf", "ga", "tournament"]].sort_values("date", kind="stable")
by_team = {t: g.reset_index(drop=True) for t, g in long.groupby("team")}


def history(team, date):
    """All of a team's internationals strictly before `date` (pre-match information only)."""
    g = by_team[team]
    return g[g.date < date]


def form10(team, date):
    h = history(team, date).tail(10)
    return h.gf.mean(), h.ga.mean()


def rest_days(team, date, cap=14):
    """Days since previous international, capped at 14 (pre-tournament camps are all >=14)."""
    return min((date - history(team, date).date.iloc[-1]).days, cap)


def h2h_gd(team, opp, date, n=10):
    h = history(team, date)
    h = h[h.opp == opp].tail(n)
    return 0.0 if h.empty else (h.gf - h.ga).mean()


def wc_matches(team, date):
    h = history(team, date)
    return int((h.tournament == "FIFA World Cup").sum())


def host_flag(team, venue):
    """1 if a host nation is playing inside its own country."""
    return int(team in HOSTS and VENUE_COUNTRY.get(venue, "United States") == team)

# ---------------------------------------------------------------------------
# 4. In-tournament running state (updated AFTER each match, read BEFORE the next)
# ---------------------------------------------------------------------------
wc = wc.sort_values(["date", "match_id"]).reset_index(drop=True)
gf_so_far, ga_so_far, last_venue = {}, {}, {}
team_rows, match_rows = [], []

for m in wc.itertuples(index=False):
    side = {}
    for team, opp, g_for, g_against in [(m.team1, m.team2, m.goals1, m.goals2),
                                        (m.team2, m.team1, m.goals2, m.goals1)]:
        f_gf, f_ga = form10(team, m.date)
        prev = last_venue.get(team)
        side[team] = dict(
            elo=elo_before(m, team), host=host_flag(team, m.venue), rest=rest_days(team, m.date),
            form_gf=f_gf, form_ga=f_ga, wc_exp=wc_matches(team, m.date),
            t_gf=gf_so_far.get(team, 0), t_ga=ga_so_far.get(team, 0),
            travel=0.0 if prev is None else haversine_km(VENUES[prev], VENUES[m.venue]),
        )
    a, b = side[m.team1], side[m.team2]

    # ---- Regression 2.1: one row per match, all predictors are team1 minus team2 ----
    match_rows.append({
        "match_id": m.match_id, "date": m.date.date(), "stage": m.stage, "team1": m.team1, "team2": m.team2,
        "elo_diff": round(a["elo"] - b["elo"], 1),                                        # shared
        "host_diff": a["host"] - b["host"],                                                # shared
        "rest_days_diff": a["rest"] - b["rest"],                                           # shared
        "form_gd_diff": round((a["form_gf"] - a["form_ga"]) - (b["form_gf"] - b["form_ga"]), 3),
        "h2h_gd": round(h2h_gd(m.team1, m.team2, m.date), 3),
        "wc_exp_diff": a["wc_exp"] - b["wc_exp"],
        "tourn_gd_diff": (a["t_gf"] - a["t_ga"]) - (b["t_gf"] - b["t_ga"]),
        "travel_km_diff": round(a["travel"] - b["travel"], 0),
        "goal_diff": m.goals1 - m.goals2,                                                 # target
    })

    # ---- Regression 2.2: one row per team per match ----
    for team, opp, g_for in [(m.team1, m.team2, m.goals1), (m.team2, m.team1, m.goals2)]:
        s, o = side[team], side[opp]
        team_rows.append({
            "row_id": f"{m.match_id}_{'A' if team == m.team1 else 'B'}", "match_id": m.match_id,
            "date": m.date.date(), "stage": m.stage, "team": team, "opponent": opp,
            "elo_diff": round(s["elo"] - o["elo"], 1),                                    # shared
            "is_host": s["host"],                                                          # shared
            "rest_days": s["rest"],                                                        # shared
            "attack_form10": round(s["form_gf"], 3),
            "opp_defence_form10": round(o["form_ga"], 3),
            "tourn_goals_so_far": s["t_gf"],
            "opp_tourn_conceded_so_far": o["t_ga"],
            "knockout": m.knockout,
            "goals": g_for,                                                                # target
        })

    # update running tournament state only AFTER the match has been recorded
    for team, gfor, gag in [(m.team1, m.goals1, m.goals2), (m.team2, m.goals2, m.goals1)]:
        gf_so_far[team] = gf_so_far.get(team, 0) + gfor
        ga_so_far[team] = ga_so_far.get(team, 0) + gag
        last_venue[team] = m.venue

reg21 = pd.DataFrame(match_rows).sort_values("match_id")
reg22 = pd.DataFrame(team_rows).sort_values(["match_id", "row_id"])
assert len(reg21) == 104 and len(reg22) == 208
assert reg21.isna().sum().sum() == 0 and reg22.isna().sum().sum() == 0

X21 = ["elo_diff", "host_diff", "rest_days_diff", "form_gd_diff", "h2h_gd", "wc_exp_diff",
       "tourn_gd_diff", "travel_km_diff"]
X22 = ["elo_diff", "is_host", "rest_days", "attack_form10", "opp_defence_form10",
       "tourn_goals_so_far", "opp_tourn_conceded_so_far", "knockout"]
assert len(X21) == 8 and len(X22) == 8

reg21.to_csv(OUT / "reg21_dataset.csv", index=False)
reg22.to_csv(OUT / "reg22_dataset.csv", index=False)

# Snapshot of pre-tournament Elo for sanity checking
pre = {t: elo_before(wc[(wc.team1 == t) | (wc.team2 == t)].iloc[0], t) for t in set(wc.team1) | set(wc.team2)}
pd.Series(pre, name="pre_tournament_elo").sort_values(ascending=False).round(0).to_csv(
    OUT / "pre_tournament_elo.csv", header=True)
print("Top-8 pre-tournament Elo:", pd.Series(pre).sort_values(ascending=False).head(8).round(0).to_dict())
print(f"reg21: {reg21.shape}  reg22: {reg22.shape}")
