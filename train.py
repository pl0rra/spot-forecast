import json, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")

TEST_DAYS, BLOCK_DAYS = 365, 30
LEADS = list(range(2, 8))
H, D = 3600, 86400
WX = ["wind_se3", "wind_se_north", "wind_dk", "wind_de", "temp_sthlm"]

pr = pd.read_csv("data/prices.csv").set_index("ts").sort_index()
wx = pd.read_csv("data/weather.csv").set_index("ts").sort_index()
t0, t1 = int(pr.index.min()), int(pr.index.max())
idx = np.arange(t0, t1 + H, H)
pr, wx = pr.reindex(idx), wx.reindex(idx)
se3, de = pr["se3"].to_numpy(float), pr["de_lu"].to_numpy(float)
n = len(idx)
i = np.arange(n)

def roll(a, w):
    return pd.Series(a).rolling(w, min_periods=w // 2).mean().to_numpy()
r24, r168, d24, d168 = roll(se3, 24), roll(se3, 168), roll(de, 24), roll(de, 168)

def take(arr, pos):
    pos = np.asarray(pos)
    out = np.full(pos.shape, np.nan)
    ok = (pos >= 0) & (pos < len(arr))
    out[ok] = arr[pos[ok]]
    return out

lags = np.column_stack([take(se3, i - 168 * k) for k in (1, 2, 3, 4)])
base = np.nanmean(lags, axis=1)
loc = pd.to_datetime(idx, unit="s", utc=True).tz_convert("Europe/Stockholm")

def build(L):
    day0 = idx - idx % D
    T = day0 - L * D
    kpos = (T + 2 * D - H - t0) // H          # sista kända timmen
    m0 = np.maximum(1, (i - kpos + 23) // 24)
    prof = np.column_stack([take(se3, i - (m0 + j) * 24) for j in range(7)])
    f = pd.DataFrame({
        "ts": idx, "T": T, "lead": L, "y": se3, "base": base,
        "p_w1": lags[:, 0], "p_w2": lags[:, 1], "p_w3": lags[:, 2], "p_w4": lags[:, 3],
        "daily7": np.nanmean(prof, axis=1),
        "mean24": take(r24, kpos), "mean168": take(r168, kpos),
        "de_mean24": take(d24, kpos), "de_mean168": take(d168, kpos),
        "de_w1": take(de, i - 168),
        "hour": loc.hour.to_numpy(), "dow": loc.dayofweek.to_numpy(), "month": loc.month.to_numpy(),
    })
    for name in WX:
        col = wx[f"{name}_d{L}"]
        f[name] = col.to_numpy(float)
        f[name + "_m24"] = col.rolling(24, min_periods=12, center=True).mean().to_numpy()
    return f

df = pd.concat([build(L) for L in LEADS], ignore_index=True)
df["res"] = df["y"] - df["base"]
df = df.dropna(subset=["y", "base"]).reset_index(drop=True)
feats = [c for c in df.columns if c not in ("ts", "T", "y", "base", "res")]

today0 = int(pd.Timestamp.now(tz="UTC").normalize().timestamp())
first_T, last_T = today0 - (TEST_DAYS + 8) * D, today0 - 8 * D
test = df[(df["T"] >= first_T) & (df["T"] <= last_T)].copy()
test["blk"] = (test["T"] - first_T) // (BLOCK_DAYS * D)
test["pred"] = np.nan

params = dict(objective="regression_l1", n_estimators=500, learning_rate=0.03,
              num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=5.0, n_jobs=2, verbose=-1)

for b in sorted(test["blk"].unique()):
    bs = first_T + int(b) * BLOCK_DAYS * D
    tr = df[df["ts"] < bs]
    model = lgb.LGBMRegressor(**params).fit(tr[feats], tr["res"])
    rows = test.index[test["blk"] == b]
    test.loc[rows, "pred"] = test.loc[rows, "base"] + model.predict(df.loc[rows, feats])
    print(f"Block {int(b)}: träning {len(tr)} rader, test {len(rows)} rader", flush=True)

test["e_base"] = (test["base"] - test["y"]).abs()
test["e_model"] = (test["pred"] - test["y"]).abs()
test["mån"] = pd.to_datetime(test["ts"], unit="s", utc=True).dt.strftime("%Y-%m")
print(test.groupby("mån")[["e_base", "e_model"]].mean().round(1))
print("ledtid  baslinje  modell")
res = {"per_lead": {}}
for L in LEADS:
    s = test[test["lead"] == L]
    res["per_lead"][L] = [round(s.e_base.mean(), 1), round(s.e_model.mean(), 1)]
    print(f"d{L}      {s.e_base.mean():6.1f}  {s.e_model.mean():6.1f}")
mb, mm = test.e_base.mean(), test.e_model.mean()
res.update(base=round(mb, 1), model=round(mm, 1))
print(f"TOTALT {mb:.1f} -> {mm:.1f} EUR/MWh ({(1 - mm / mb) * 100:.0f}% bättre)")

imp = sorted(zip(feats, model.booster_.feature_importance("gain")), key=lambda x: -x[1])[:15]
print("Viktigaste indata (sista modellen):")
for k, v in imp:
    print(f"  {k:18s} {v:12.0f}")
res["importance"] = {k: round(float(v)) for k, v in imp}
with open("train_results.json", "w") as f:
    json.dump(res, f, indent=1)
