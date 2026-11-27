import json, os, time, datetime as dt, urllib.request, urllib.parse, urllib.error
import warnings
from collections import defaultdict
from statistics import mean
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")

UTC = dt.timezone.utc
H, D = 3600, 86400
LEADS = list(range(2, 8))
LAMBDA, RECENT_DAYS = 0.5, 30
POINTS = {
    "wind_se3": [(58.5, 13.0), (57.8, 15.2), (57.1, 18.1), (59.7, 14.0)],
    "wind_se_north": [(63.2, 17.3), (65.5, 20.7)],
    "wind_dk": [(56.0, 8.4)],
    "wind_de": [(54.2, 7.0), (53.5, 9.0)],
    "temp_sthlm": [(59.33, 18.07)],
}
WX = list(POINTS)
now = dt.datetime.now(UTC)
today = now.date()
today0 = int(dt.datetime(now.year, now.month, now.day, tzinfo=UTC).timestamp())

def get(url, params, tries=4):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": "spot-forecast/1.0"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'ignore')}", flush=True)
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(30 * (i + 1))
                continue
            return None
        except Exception as e:
            print("Fel:", e, flush=True)
            time.sleep(10)
    return None

# ---------- data ----------
pr = pd.read_csv("data/prices.csv").set_index("ts").sort_index()
for col, bzn in (("se3", "SE3"), ("de_lu", "DE-LU")):
    d = get("https://api.energy-charts.info/price", {
        "bzn": bzn, "start": (today - dt.timedelta(days=10)).isoformat(),
        "end": (today + dt.timedelta(days=2)).isoformat()})
    if d and "unix_seconds" in d:
        by = defaultdict(list)
        for t, p in zip(d["unix_seconds"], d["price"]):
            if p is not None:
                by[t // 3600 * 3600].append(p)
        for h, v in by.items():
            pr.loc[h, col] = round(mean(v), 3)
        print("Färska priser", bzn, len(by), "timmar", flush=True)
    time.sleep(3)
pr = pr.sort_index()

t0 = int(pr.index.min())
idx = np.arange(t0, today0 + 8 * D, H)
pr = pr.reindex(idx)
wx = pd.read_csv("data/weather.csv").set_index("ts").sort_index().reindex(idx)
se3, de = pr["se3"].to_numpy(float), pr["de_lu"].to_numpy(float)
n = len(idx)
i = np.arange(n)

def ts_of(t):
    return int(dt.datetime.fromisoformat(t).replace(tzinfo=UTC).timestamp())

def live_weather():
    out = {}
    for name, pts in POINTS.items():
        var = "temperature_2m" if name.startswith("temp") else "wind_speed_120m"
        agg = defaultdict(list)
        for lat, lon in pts:
            d = get("https://api.open-meteo.com/v1/forecast", {
                "latitude": lat, "longitude": lon, "hourly": var, "wind_speed_unit": "ms",
                "past_days": 2, "forecast_days": 9, "timezone": "UTC"})
            if d and "hourly" in d:
                for t, v in zip(d["hourly"]["time"], d["hourly"][var]):
                    if v is not None:
                        agg[ts_of(t)].append(v)
            time.sleep(2)
        out[name] = {t: mean(v) for t, v in agg.items()}
        print("Prognos", name, len(out[name]), "timmar", flush=True)
    return out

live = live_weather()
for name in WX:
    s = pd.Series(live.get(name, {}), dtype=float).reindex(idx).to_numpy()
    for L in LEADS:
        c = f"{name}_d{L}"
        wx[c] = wx[c].where(idx < today0, s)

# ---------- indata ----------
def roll(a, w):
    return pd.Series(a).rolling(w, min_periods=w // 2).mean().to_numpy()

def take(arr, pos):
    pos = np.asarray(pos)
    out = np.full(pos.shape, np.nan)
    ok = (pos >= 0) & (pos < len(arr))
    out[ok] = arr[pos[ok]]
    return out

r24, r168, d24, d168 = roll(se3, 24), roll(se3, 168), roll(de, 24), roll(de, 168)
nu = pd.read_csv("data/nuclear.csv").set_index("ts").reindex(idx)["nuclear_mw"].to_numpy(float)
nu24, nu168 = roll(nu, 24), roll(nu, 168)

hw = pd.read_csv("data/hydro.csv")
hw["total"] = hw[["no1", "no2", "no3", "no4", "no5"]].sum(axis=1, min_count=5)
hw["week"] = pd.to_datetime(hw["ts"], unit="s", utc=True).dt.isocalendar().week.astype(int)
hw["anom"] = (hw["total"] / hw.groupby("week")["total"].transform("median") - 1) * 100
hs = hw.set_index(hw["ts"] + 10 * D)["anom"].dropna()
hs = hs[~hs.index.duplicated()]
hy = hs.reindex(hs.index.union(idx)).sort_index().ffill().reindex(idx).to_numpy(float)

lags = np.column_stack([take(se3, i - 168 * k) for k in (1, 2, 3, 4)])
base = np.nanmean(lags, axis=1)
rb24, rb168 = roll(base, 24), roll(base, 168)
loc = pd.to_datetime(idx, unit="s", utc=True).tz_convert("Europe/Stockholm")

def build(L):
    day0 = idx - idx % D
    T = day0 - L * D
    kpos = (T + 2 * D - H - t0) // H
    gpos = (T + 11 * H - t0) // H
    m0 = np.maximum(1, (i - kpos + 23) // 24)
    prof = np.column_stack([take(se3, i - (m0 + j) * 24) for j in range(7)])
    f = pd.DataFrame({
        "ts": idx, "T": T, "lead": L, "y": se3, "base": base,
        "p_w1": lags[:, 0], "p_w2": lags[:, 1], "p_w3": lags[:, 2], "p_w4": lags[:, 3],
        "daily7": np.nanmean(prof, axis=1),
        "mean24": take(r24, kpos), "mean168": take(r168, kpos),
        "de_mean24": take(d24, kpos), "de_mean168": take(d168, kpos),
        "de_w1": take(de, i - 168),
        "dev24": take(r24, kpos) - take(rb24, kpos),
        "dev168": take(r168, kpos) - take(rb168, kpos),
        "spread24": take(d24, kpos) - take(r24, kpos),
        "spread168": take(d168, kpos) - take(r168, kpos),
        "nuc24": take(nu24, gpos), "nuc168": take(nu168, gpos),
        "nuc_chg": take(nu24, gpos) - take(nu168, gpos),
        "hydro_anom": take(hy, gpos),
        "hour": loc.hour.to_numpy(), "dow": loc.dayofweek.to_numpy(),
    })
    for name in WX:
        col = wx[f"{name}_d{L}"]
        f[name] = col.to_numpy(float)
        f[name + "_m24"] = col.rolling(24, min_periods=12, center=True).mean().to_numpy()
    return f

df = pd.concat([build(L) for L in LEADS], ignore_index=True)
feats = [c for c in df.columns if c not in ("ts", "T", "y", "base")]
known = df[df["y"].notna() & df["base"].notna()]

params = dict(objective="regression_l1", n_estimators=500, learning_rate=0.03,
              num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=5.0, n_jobs=2, verbose=-1)

# ---------- modell A: senaste dagarnas fel (bias och osäkerhet) ----------
cut = today0 - RECENT_DAYS * D
trA = known[known["ts"] < cut]
recent = known[(known["T"] >= cut) & (known["T"] <= today0 - D)].copy()
mA = lgb.LGBMRegressor(**params).fit(trA[feats], trA["y"] - trA["base"])
recent["pred"] = recent["base"] + mA.predict(recent[feats])
recent["err"] = recent["pred"] - recent["y"]
kp = today0 + 2 * D - H
win = recent[(recent["ts"] > kp - 14 * D) & (recent["ts"] <= kp)]
bias_hat = float(win["err"].mean()) if len(win) >= 300 else 0.0
q = {}
for L in LEADS:
    r = -recent.loc[recent["lead"] == L, "err"]
    q[L] = ((float(r.quantile(0.1) - r.quantile(0.5)), float(r.quantile(0.9) - r.quantile(0.5)))
            if len(r) >= 100 else (-30.0, 30.0))
print(f"Senaste {RECENT_DAYS} d: modell MAE {recent['err'].abs().mean():.1f}, "
      f"baslinje {(recent['base'] - recent['y']).abs().mean():.1f}, bias_hat {bias_hat:.1f}", flush=True)

# ---------- modell B: all data, prognos ----------
trB = known[known["ts"] < today0]
mB = lgb.LGBMRegressor(**params).fit(trB[feats], trB["y"] - trB["base"])
liv = df[(df["T"] == today0) & (df["ts"] >= today0 + 2 * D)].copy()
if liv.empty:
    raise SystemExit("Inga rader att prognostisera")
liv["pred"] = liv["base"] + mB.predict(liv[feats]) - LAMBDA * bias_hat
liv["lo"] = liv["pred"] + liv["lead"].map(lambda L: q[L][0])
liv["hi"] = liv["pred"] + liv["lead"].map(lambda L: q[L][1])
lp = liv.set_index("ts")

# ---------- skriv ----------
from zoneinfo import ZoneInfo
SE = ZoneInfo("Europe/Stockholm")
start = int(now.astimezone(SE).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
times, prices, kn, los, his = [], [], [], [], []
for h in range(start, start + 192 * H, H):
    pos = (h - t0) // H
    y = se3[pos]
    if not np.isnan(y):
        val, k, a, b = y, True, y, y
    elif h in lp.index:
        val, k, a, b = lp.at[h, "pred"], False, lp.at[h, "lo"], lp.at[h, "hi"]
    elif not np.isnan(base[pos]):
        val, k, a, b = base[pos], False, base[pos] - 30, base[pos] + 30
    else:
        continue
    times.append(dt.datetime.fromtimestamp(h, UTC).isoformat())
    prices.append(round(float(val), 2)); kn.append(bool(k))
    los.append(round(float(a), 2)); his.append(round(float(b), 2))

from zoneinfo import ZoneInfo
SE = ZoneInfo("Europe/Stockholm")
WINDOW, MOMS = 3, 1.0          # MOMS = 1.25 ger pris inklusive moms

def eur_sek():
    for params in ({"base": "EUR", "symbols": "SEK"}, {"from": "EUR", "to": "SEK"}):
        d = get("https://api.frankfurter.dev/v1/latest", params, tries=2)
        try:
            return float(d["rates"]["SEK"])
        except Exception:
            pass
    print("Varning: ingen växelkurs hittades, använder 11.0", flush=True)
    return 11.0

RATE = eur_sek()

def ore(eur_mwh):
    return eur_mwh * RATE / 10 * MOMS

def label(nk, n):
    return "Nord Pool" if nk == n else ("prognos" if nk == 0 else "delvis prognos")

def day_windows(times, vals, kn):
    days = defaultdict(list)
    for t, v, k in zip(times, vals, kn):
        u = dt.datetime.fromisoformat(t)
        days[u.astimezone(SE).date()].append((u, v, k))
    res = []
    for day, rows in sorted(days.items()):
        rows.sort(key=lambda r: r[0])
        if len(rows) < 12:
            continue
        res.append({"day": day.isoformat(), "kind": "snitt",
                    "avg_ore": round(ore(mean(r[1] for r in rows)), 1),
                    "status": label(sum(1 for r in rows if r[2]), len(rows))})
        cands = []
        for j in range(len(rows) - WINDOW + 1):
            w = rows[j:j + WINDOW]
            if w[-1][0] - w[0][0] == dt.timedelta(hours=WINDOW - 1):
                cands.append((mean(r[1] for r in w), w))
        if not cands:
            continue
        for kind, (m, w) in (("billigast", min(cands, key=lambda c: c[0])),
                             ("dyrast", max(cands, key=lambda c: c[0]))):
            res.append({"day": day.isoformat(), "kind": kind,
                        "start": w[0][0].isoformat(),
                        "end": (w[-1][0] + dt.timedelta(hours=1)).isoformat(),
                        "avg_ore": round(ore(m), 1),
                        "status": label(sum(1 for r in w if r[2]), WINDOW)})
    return res

wins = day_windows(times, prices, kn)

out = {"generated": now.isoformat(timespec="seconds"), "zone": "SE3", "unit": "EUR/MWh",
       "method": f"lgbm_residual_bias{LAMBDA}", "bias_hat": round(bias_hat, 1),
       "rate": round(RATE, 4), "windows": wins,
       "time": times, "price": prices, "known": kn, "lo": los, "hi": his}
os.makedirs("history", exist_ok=True)
for path in ("forecast.json", f"history/{today.isoformat()}.json"):
    with open(path, "w") as f:
        json.dump(out, f)

# ---------- kalenderfil ----------
def esc(s):
    return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")

def z(u):
    return u.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")

stamp = z(now)
cal = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//spot-forecast//SE3//SV", "CALSCALE:GREGORIAN"]
for w in wins:
    pre = "" if w["status"] == "Nord Pool" else "≈ "
    cal += ["BEGIN:VEVENT", f"UID:{w['kind']}-{w['day']}@spot-forecast", f"DTSTAMP:{stamp}"]
    if w["kind"] == "snitt":
        d0 = dt.date.fromisoformat(w["day"])
        summary = f"Snitt {pre}{w['avg_ore']:.0f} öre/kWh ({w['status']})"
        cal += [f"DTSTART;VALUE=DATE:{d0:%Y%m%d}",
                f"DTEND;VALUE=DATE:{d0 + dt.timedelta(days=1):%Y%m%d}"]
    else:
        s0 = dt.datetime.fromisoformat(w["start"])
        e0 = dt.datetime.fromisoformat(w["end"])
        summary = (f"{w['kind'].capitalize()} {s0.astimezone(SE):%H}–{e0.astimezone(SE):%H} "
                   f"{pre}{w['avg_ore']:.0f} öre ({w['status']})")
        cal += [f"DTSTART:{z(s0)}", f"DTEND:{z(e0)}"]
    cal += [f"SUMMARY:{esc(summary)}", "END:VEVENT"]
cal.append("END:VCALENDAR")
with open("prisprognos.ics", "w", encoding="utf-8", newline="") as f:
    f.write("\r\n".join(cal) + "\r\n")
print("OK", len(times), "timmar,", kn.count(True), "kända, bias_hat", round(bias_hat, 1),
      "kurs", round(RATE, 3), "händelser", len(wins))
