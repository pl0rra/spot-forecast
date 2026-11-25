import csv, os, time, datetime as dt, urllib.request, urllib.parse, urllib.error
from collections import defaultdict
from statistics import mean

UTC = dt.timezone.utc
DAYS_HISTORY, OVERLAP_DAYS = 730, 3
PRICE_CHUNK, WX_CHUNK = 30, 180
LEADS = range(2, 8)
SUFFIXES = ["obs"] + [f"d{k}" for k in LEADS]
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
POINTS = {
    "wind_se3": [(58.5, 13.0), (57.8, 15.2), (57.1, 18.1), (59.7, 14.0)],
    "wind_se_north": [(63.2, 17.3), (65.5, 20.7)],
    "wind_dk": [(56.0, 8.4)],
    "wind_de": [(54.2, 7.0), (53.5, 9.0)],
    "temp_sthlm": [(59.33, 18.07)],
}
today = dt.datetime.now(UTC).date()
os.makedirs("data", exist_ok=True)

def get(url, params, tries=5):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": "spot-forecast/1.0"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                import json
                return json.load(r)
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'ignore')}", flush=True)
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(30 * (i + 1))
                continue
            return None
        except Exception as e:
            print("Fel:", e, flush=True)
            time.sleep(10)
    return None

def read_csv(path):
    rows = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[int(r["ts"])] = r
    return rows

def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts"] + cols)
        w.writeheader()
        for ts in sorted(rows):
            w.writerow({"ts": ts, **{c: rows[ts].get(c, "") for c in cols}})

def start_for(rows):
    first = today - dt.timedelta(days=DAYS_HISTORY)
    if rows:
        last = dt.datetime.fromtimestamp(max(rows), UTC).date()
        return max(last - dt.timedelta(days=OVERLAP_DAYS), first)
    return first

def chunks(start, end, n):
    d0 = start
    while d0 <= end:
        d1 = min(d0 + dt.timedelta(days=n - 1), end)
        yield d0, d1
        d0 = d1 + dt.timedelta(days=1)

def update_prices():
    path = "data/prices.csv"
    rows = read_csv(path)
    start = start_for(rows)
    zones = {"se3": "SE3", "de_lu": "DE-LU"}
    for col, bzn in zones.items():
        by_hour = defaultdict(list)
        for d0, d1 in chunks(start, today, PRICE_CHUNK):
            d = get("https://api.energy-charts.info/price",
                    {"bzn": bzn, "start": d0.isoformat(), "end": d1.isoformat()})
            if d and "unix_seconds" in d:
                for t, p in zip(d["unix_seconds"], d["price"]):
                    if p is not None:
                        by_hour[t // 3600 * 3600].append(p)
                print("Pris", bzn, d0, d1, flush=True)
            else:
                print("Hoppar över pris", bzn, d0, d1, flush=True)
            time.sleep(10)
        for h, v in by_hour.items():
            rows.setdefault(h, {"ts": str(h)})[col] = round(mean(v), 3)
    write_csv(path, rows, list(zones))

def pick_wind_var():
    s = (today - dt.timedelta(days=5)).isoformat()
    e = (today - dt.timedelta(days=2)).isoformat()
    for v in ("wind_speed_120m", "wind_speed_10m"):
        d = get(URL, {"latitude": 59.7, "longitude": 14.0, "start_date": s, "end_date": e,
                      "hourly": f"{v}_previous_day2", "timezone": "UTC"})
        if d and f"{v}_previous_day2" in d.get("hourly", {}):
            print("Vindvariabel:", v, flush=True)
            return v
        time.sleep(5)
    raise SystemExit("Ingen fungerande vindvariabel i Previous Runs API")

def fetch_point(lat, lon, var, start):
    names = [var] + [f"{var}_previous_day{k}" for k in LEADS]
    out = {n: {} for n in names}
    for d0, d1 in chunks(start, today, WX_CHUNK):
        d = get(URL, {"latitude": lat, "longitude": lon, "start_date": d0.isoformat(),
                      "end_date": d1.isoformat(), "hourly": ",".join(names),
                      "wind_speed_unit": "ms", "timezone": "UTC"})
        if d and "hourly" in d:
            times = d["hourly"]["time"]
            for n in names:
                for t, v in zip(times, d["hourly"].get(n, [])):
                    if v is not None:
                        ts = int(dt.datetime.fromisoformat(t).replace(tzinfo=UTC).timestamp())
                        out[n][ts] = v
            print("Väder", lat, lon, d0, d1, flush=True)
        else:
            print("Hoppar över väder", lat, lon, d0, d1, flush=True)
        time.sleep(5)
    return out

def update_weather():
    path = "data/weather.csv"
    rows = read_csv(path)
    start = start_for(rows)
    wind_var = pick_wind_var()
    cols = [f"{n}_{s}" for n in POINTS for s in SUFFIXES]
    for name, pts in POINTS.items():
        var = "temperature_2m" if name.startswith("temp") else wind_var
        agg = defaultdict(lambda: defaultdict(list))
        for lat, lon in pts:
            for n, vals in fetch_point(lat, lon, var, start).items():
                suffix = "obs" if n == var else "d" + n.rsplit("previous_day", 1)[1]
                for ts, v in vals.items():
                    agg[suffix][ts].append(v)
        for suffix, byts in agg.items():
            for ts, vs in byts.items():
                rows.setdefault(ts, {"ts": str(ts)})[f"{name}_{suffix}"] = round(mean(vs), 2)
    write_csv(path, rows, cols)

update_prices()
update_weather()