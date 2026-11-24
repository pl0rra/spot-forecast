import json, datetime as dt, urllib.request, urllib.parse
from collections import defaultdict
from statistics import mean, median

BZN = "SE3"
DAYS_HISTORY = 730
TEST_DAYS = 180
CHUNK = 90
DAY, W = 86400, 7 * 86400

def get(url, params):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": "spot-forecast/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def fetch_prices(bzn, start, end):
    by_hour = defaultdict(list)
    d0 = start
    while d0 <= end:
        d1 = min(d0 + dt.timedelta(days=CHUNK - 1), end)
        d = get("https://api.energy-charts.info/price",
                {"bzn": bzn, "start": d0.isoformat(), "end": d1.isoformat()})
        for t, p in zip(d["unix_seconds"], d["price"]):
            if p is not None:
                by_hour[t // 3600 * 3600].append(p)
        d0 = d1 + dt.timedelta(days=1)
    return {h: mean(v) for h, v in by_hour.items()}

today = dt.datetime.now(dt.timezone.utc).date()
hourly = fetch_prices(BZN, today - dt.timedelta(days=DAYS_HISTORY), today)
print("Timmar hämtade:", len(hourly))

def weeks(h, ku, agg):
    v = [hourly[h - w * W] for w in (1, 2, 3, 4)
         if (h - w * W) in hourly and (h - w * W) <= ku]
    return agg(v) if v else None

def one_week(h, ku):
    x = h - W
    return hourly.get(x) if x <= ku else None

def daily7(h, ku):
    k0 = max(1, (h - ku + DAY - 1) // DAY)
    v = [hourly[h - (k0 + j) * DAY] for j in range(7) if (h - (k0 + j) * DAY) in hourly]
    return mean(v) if v else None

def flat7(h, ku):
    v = [hourly[x] for x in range(ku - 167 * 3600, ku + 1, 3600) if x in hourly]
    return mean(v) if v else None

models = {
    "4v_medel": lambda h, ku: weeks(h, ku, mean),
    "4v_median": lambda h, ku: weeks(h, ku, median),
    "1v_sen": one_week,
    "7d_dygnsprofil": daily7,
    "platt_7d": flat7,
}
errs = {m: defaultdict(list) for m in models}
actuals = []

for back in range(TEST_DAYS + 8, 7, -1):
    T = today - dt.timedelta(days=back)
    tstart = int(dt.datetime(T.year, T.month, T.day, tzinfo=dt.timezone.utc).timestamp())
    ku = tstart + 2 * DAY - 3600
    for h in range(tstart + 2 * DAY, tstart + 8 * DAY, 3600):
        if h not in hourly:
            continue
        lead = (h - tstart) // DAY
        actuals.append(hourly[h])
        for name, f in models.items():
            p = f(h, ku)
            if p is not None:
                errs[name][lead].append(abs(p - hourly[h]))

avg = mean(actuals)
print(f"Testperiod {TEST_DAYS} dagar, medelpris {avg:.1f} EUR/MWh, {len(actuals)} timmar")
result = {"zone": BZN, "test_days": TEST_DAYS, "mean_price": round(avg, 2), "mae": {}}
for name, d in errs.items():
    allv = [e for v in d.values() for e in v]
    per = {k: round(mean(d[k]), 1) for k in sorted(d)}
    result["mae"][name] = {"total": round(mean(allv), 1), "per_ledtid_dag": per}
    print(f"{name:16s} MAE {mean(allv):6.1f}   " + "  ".join(f"d{k}:{v:5.1f}" for k, v in per.items()))

with open("backtest.json", "w") as f:
    json.dump(result, f, indent=1)
