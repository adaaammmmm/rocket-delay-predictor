"""
generate_data.py
Generates a realistic synthetic rocket launch dataset and saves it to raw/launches.csv
Run: python data/generate_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
N_LAUNCHES = 4600

COMPANIES = {
    "SpaceX":          {"base_success": 0.95, "founded": 2002},
    "Roscosmos":       {"base_success": 0.88, "founded": 1992},
    "ULA":             {"base_success": 0.97, "founded": 2006},
    "Arianespace":     {"base_success": 0.94, "founded": 1980},
    "ISRO":            {"base_success": 0.85, "founded": 1969},
    "JAXA":            {"base_success": 0.90, "founded": 2003},
    "Blue Origin":     {"base_success": 0.80, "founded": 2000},
    "Rocket Lab":      {"base_success": 0.87, "founded": 2006},
    "Northrop Grumman":{"base_success": 0.91, "founded": 1994},
    "Virgin Orbit":    {"base_success": 0.60, "founded": 2017},
}

ROCKETS = {
    "Falcon 9":        {"company": "SpaceX",           "first_flight": 2010},
    "Falcon Heavy":    {"company": "SpaceX",           "first_flight": 2018},
    "Soyuz":           {"company": "Roscosmos",        "first_flight": 1966},
    "Atlas V":         {"company": "ULA",              "first_flight": 2002},
    "Vulcan":          {"company": "ULA",              "first_flight": 2024},
    "Ariane 5":        {"company": "Arianespace",      "first_flight": 1996},
    "Ariane 6":        {"company": "Arianespace",      "first_flight": 2024},
    "PSLV":            {"company": "ISRO",             "first_flight": 1993},
    "H-IIA":           {"company": "JAXA",             "first_flight": 2001},
    "New Shepard":     {"company": "Blue Origin",      "first_flight": 2015},
    "Electron":        {"company": "Rocket Lab",       "first_flight": 2017},
    "Antares":         {"company": "Northrop Grumman", "first_flight": 2013},
    "LauncherOne":     {"company": "Virgin Orbit",     "first_flight": 2020},
}

SITES = {
    "Cape Canaveral, FL": {"lat": 28.4, "lon": -80.6, "country": "USA"},
    "Vandenberg, CA":     {"lat": 34.7, "lon": -120.6,"country": "USA"},
    "Baikonur, Kazakhstan":{"lat": 45.9,"lon": 63.3,  "country": "Kazakhstan"},
    "Kourou, French Guiana":{"lat": 5.2,"lon": -52.8, "country": "France"},
    "Satish Dhawan, India":{"lat": 13.7,"lon": 80.2,  "country": "India"},
    "Tanegashima, Japan":  {"lat": 30.4,"lon": 130.9, "country": "Japan"},
    "Wallops Island, VA":  {"lat": 37.9,"lon": -75.5, "country": "USA"},
    "Mahia, New Zealand":  {"lat": -39.3,"lon": 177.9,"country": "New Zealand"},
}

MISSION_TYPES = ["Satellite", "ISS Resupply", "Science", "Military", "Commercial", "Crewed", "Test"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def random_date(start_year=1990, end_year=2024):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def simulate_weather(site_name, launch_date):
    """Simulate plausible weather for a site and date."""
    lat = SITES[site_name]["lat"]
    month = launch_date.month
    # Seasonal temperature
    if lat > 0:  # Northern hemisphere
        base_temp = 15 + 15 * np.sin((month - 4) * np.pi / 6)
    else:
        base_temp = 20 - 15 * np.sin((month - 4) * np.pi / 6)

    temp          = round(base_temp + np.random.normal(0, 5), 1)
    wind_speed    = round(abs(np.random.normal(18, 12)), 1)
    precipitation = round(max(0, np.random.exponential(2)), 2)
    cloud_cover   = round(min(100, max(0, np.random.normal(40, 25))), 1)
    return temp, wind_speed, precipitation, cloud_cover


def compute_delay_prob(company, rocket_name, launch_date, wind_speed, precipitation,
                       cloud_cover, provider_hist_rate, site_hist_rate, is_crewed):
    """Simulate delay probability from realistic rules."""
    base      = 1 - COMPANIES[company]["base_success"]
    age_years = (launch_date.year - ROCKETS[rocket_name]["first_flight"])
    age_pen   = max(0, (age_years - 15) * 0.005)   # older rockets slightly riskier

    wind_pen  = max(0, (wind_speed - 35) * 0.006)
    rain_pen  = min(0.2, precipitation * 0.04)
    cloud_pen = max(0, (cloud_cover - 80) * 0.002)
    hist_pen  = (1 - provider_hist_rate) * 0.3
    site_pen  = (1 - site_hist_rate) * 0.2
    crew_pen  = 0.05 if is_crewed else 0.0           # extra scrutiny for crewed

    prob = base + age_pen + wind_pen + rain_pen + cloud_pen + hist_pen + site_pen + crew_pen
    return min(0.9, max(0.02, prob))


# ── Main generation loop ─────────────────────────────────────────────────────

rows = []
provider_outcomes: dict[str, list] = {c: [] for c in COMPANIES}
site_outcomes:     dict[str, list] = {s: [] for s in SITES}

for i in range(N_LAUNCHES):
    rocket_name = random.choice(list(ROCKETS.keys()))
    company     = ROCKETS[rocket_name]["company"]
    site_name   = random.choice(list(SITES.keys()))
    launch_date = random_date()
    mission_type= random.choice(MISSION_TYPES)
    is_crewed   = int(mission_type == "Crewed")

    temp, wind_speed, precipitation, cloud_cover = simulate_weather(site_name, launch_date)

    # Rolling historical rates (use previous launches)
    prev_prov = provider_outcomes[company]
    prev_site = site_outcomes[site_name]
    prov_rate = np.mean(prev_prov[-50:]) if prev_prov else 0.90
    site_rate = np.mean(prev_site[-50:]) if prev_site else 0.90

    age_years = launch_date.year - ROCKETS[rocket_name]["first_flight"]
    launches_in_month = sum(
        1 for r in rows
        if r["launch_year"] == launch_date.year and r["launch_month"] == launch_date.month
    )

    delay_prob = compute_delay_prob(
        company, rocket_name, launch_date, wind_speed, precipitation,
        cloud_cover, prov_rate, site_rate, is_crewed
    )

    delayed = int(np.random.random() < delay_prob)
    delay_hours = 0
    if delayed:
        delay_hours = round(max(0.5, np.random.exponential(12)), 1)

    outcome = 1 - delayed
    provider_outcomes[company].append(outcome)
    site_outcomes[site_name].append(outcome)

    rows.append({
        "launch_id":           i + 1,
        "company":             company,
        "rocket":              rocket_name,
        "site":                site_name,
        "site_country":        SITES[site_name]["country"],
        "site_lat":            SITES[site_name]["lat"],
        "site_lon":            SITES[site_name]["lon"],
        "launch_date":         launch_date.strftime("%Y-%m-%d"),
        "launch_year":         launch_date.year,
        "launch_month":        launch_date.month,
        "mission_type":        mission_type,
        "is_crewed":           is_crewed,
        "rocket_first_flight": ROCKETS[rocket_name]["first_flight"],
        "rocket_age_years":    age_years,
        "provider_success_rate": round(prov_rate, 4),
        "site_success_rate":     round(site_rate, 4),
        "launches_this_month":   launches_in_month,
        "temp_celsius":          temp,
        "wind_speed_kmh":        wind_speed,
        "precipitation_mm":      precipitation,
        "cloud_cover_pct":       cloud_cover,
        "delayed":               delayed,
        "delay_hours":           delay_hours,
    })

df = pd.DataFrame(rows)
out_path = os.path.join(os.path.dirname(__file__), "raw", "launches.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"✅ Saved {len(df)} launches → {out_path}")
print(f"   Delay rate: {df['delayed'].mean():.1%}")
print(df.head(3).to_string())
