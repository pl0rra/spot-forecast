import json, os, datetime as dt, urllib.request, urllib.parse
from collections import defaultdict
from statistics import mean

BZN = "SE3"
DAYS_BACK = 35
HORIZON_HOURS = 7 * 24

def get(url, params):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": "spot-forecast/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

now = dt.datetime.now(dt.timezone.utc)
today = now.date()
d = get("https://api.energy-charts.info/price", {
    "bzn": BZN,
    "start": (today - dt.timedelta(days=DAYS_BACK)).isoformat(),
    "end": (today + dt.timedelta(days=2)).isoformat(),
})

by_hour = defaultdict(list)
for t, p in zip(d["unix_seconds"], d["price"]):
    if p is not None:
        by_hour[t // 3600 * 3600].append(p)
hourly = {h: mean(v) for h, v in by_hour.items()}
if not hourly:
    raise SystemExit("Inga priser returnerades, kontrollera BZN")

start = int(now.timestamp()) // 3600 * 3600
times, prices, known = [], [], []
for i in range(HORIZON_HOURS):
    h = start + i * 3600
    if h in hourly:
        val, k = hourly[h], True
    else:
        past = [hourly[h - w * 7 * 86400] for w in (1, 2, 3, 4)
                if (h - w * 7 * 86400) in hourly]
        if not past:
            continue
        val, k = mean(past), False
    times.append(dt.datetime.fromtimestamp(h, dt.timezone.utc).isoformat())
    prices.append(round(val, 2))
    known.append(k)

out = {"generated": now.isoformat(timespec="seconds"), "zone": BZN,
       "unit": "EUR/MWh", "method": "baseline_same_hour_4w",
       "time": times, "price": prices, "known": known}

os.makedirs("history", exist_ok=True)
with open("forecast.json", "w") as f:
    json.dump(out, f)
with open(f"history/{today.isoformat()}.json", "w") as f:
    json.dump(out, f)
print("OK", len(times), "timmar, känt till", times[known.count(True) - 1] if any(known) else "-")
