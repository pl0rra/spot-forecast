import csv, os, time, datetime as dt, urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
from collections import defaultdict
from statistics import mean

TOKEN = os.environ["ae20322e-347f-477c-b23f-5b7a7750a3c2"] #ENTSO-token
UTC = dt.timezone.utc
BASE = "https://web-api.tp.entsoe.eu/api"
SE3 = "10Y1001A1001A46L"
NO_ZONES = {"NO1": "10YNO-1--------2", "NO2": "10YNO-2--------T", "NO3": "10YNO-3--------J",
            "NO4": "10YNO-4--------9", "NO5": "10Y1001A1001A48H"}
RES = {"PT60M": dt.timedelta(hours=1), "PT30M": dt.timedelta(minutes=30),
       "PT15M": dt.timedelta(minutes=15), "P1D": dt.timedelta(days=1), "P7D": dt.timedelta(days=7)}
today = dt.datetime.now(UTC).date()
os.makedirs("data", exist_ok=True)

def entsoe(params, tries=4):
    q = urllib.parse.urlencode({**params, "securityToken": TOKEN})
    req = urllib.request.Request(f"{BASE}?{q}", headers={"User-Agent": "spot-forecast/1.0"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'ignore')}", flush=True)
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(20 * (i + 1))
                continue
            return None
        except Exception as e:
            print("Fel:", type(e).__name__, flush=True)
            time.sleep(10)
    return None

def strip(tag):
    return tag.rsplit("}", 1)[-1]

def parse(xml_text):
    out = []
    root = ET.fromstring(xml_text)
    for ts_el in root.iter():
        if strip(ts_el.tag) != "TimeSeries":
            continue
        for per in ts_el:
            if strip(per.tag) != "Period":
                continue
            start = step = None
            pts = []
            for ch in per:
                t = strip(ch.tag)
                if t == "timeInterval":
                    for x in ch:
                        if strip(x.tag) == "start":
                            start = dt.datetime.strptime(x.text, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
                elif t == "resolution":
                    step = RES.get(ch.text)
                elif t == "Point":
                    pos = qty = None
                    for x in ch:
                        if strip(x.tag) == "position":
                            pos = int(x.text)
                        elif strip(x.tag) == "quantity":
                            qty = float(x.text)
                    if pos is not None and qty is not None:
                        pts.append((pos, qty))
            if start and step:
                for pos, qty in pts:
                    out.append((start + (pos - 1) * step, qty))
    return out

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

def start_for(rows, history_days, overlap=3):
    first = today - dt.timedelta(days=history_days)
    if rows:
        last = dt.datetime.fromtimestamp(max(rows), UTC).date()
        return max(last - dt.timedelta(days=overlap), first)
    return first

def chunks(start, end, n):
    d0 = start
    while d0 <= end:
        d1 = min(d0 + dt.timedelta(days=n - 1), end)
        yield d0, d1
        d0 = d1 + dt.timedelta(days=1)

def period(d0, d1):
    return {"periodStart": d0.strftime("%Y%m%d0000"),
            "periodEnd": (d1 + dt.timedelta(days=1)).strftime("%Y%m%d0000")}

def update_nuclear():
    path = "data/nuclear.csv"
    rows = read_csv(path)
    byh = defaultdict(list)
    for d0, d1 in chunks(start_for(rows, 730), today, 60):
        xml = entsoe({"documentType": "A75", "processType": "A16", "in_Domain": SE3,
                      "psrType": "B14", **period(d0, d1)})
        if xml:
            for ts, q in parse(xml):
                byh[int(ts.timestamp()) // 3600 * 3600].append(q)
            print("Kärnkraft", d0, d1, flush=True)
        else:
            print("Hoppar över kärnkraft", d0, d1, flush=True)
        time.sleep(2)
    for h, v in byh.items():
        rows.setdefault(h, {"ts": str(h)})["nuclear_mw"] = round(mean(v), 1)
    write_csv(path, rows, ["nuclear_mw"])

def update_hydro():
    path = "data/hydro.csv"
    rows = read_csv(path)
    start = start_for(rows, 3650, overlap=21)
    for z, eic in NO_ZONES.items():
        for d0, d1 in chunks(start, today, 365):
            xml = entsoe({"documentType": "A72", "processType": "A16",
                          "in_Domain": eic, **period(d0, d1)})
            if xml:
                for ts, q in parse(xml):
                    key = (int(ts.timestamp()) + 12 * 3600) // 86400 * 86400
                    rows.setdefault(key, {"ts": str(key)})[z.lower()] = q
                print("Magasin", z, d0, d1, flush=True)
            else:
                print("Hoppar över magasin", z, d0, d1, flush=True)
            time.sleep(2)
    write_csv(path, rows, [z.lower() for z in NO_ZONES])

update_nuclear()
update_hydro()
