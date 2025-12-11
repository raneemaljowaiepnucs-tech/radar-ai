import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np

df = pd.read_excel("absher_radar_data_en.xlsx")

service_counts = df.groupby("service")["requests"].sum().sort_values(ascending=False)
region_counts = df.groupby("region")["requests"].sum().sort_values(ascending=False)
hour_counts   = df.groupby("hour")["requests"].sum()

viol = df[df["service"] == "Violations"].copy()
viol = viol.sort_values("hour")
pattern = viol["requests"].values
series = np.tile(pattern, 365)

dates = pd.date_range(start="2025-01-01", periods=len(series), freq="H")
ts = pd.DataFrame({"datetime": dates, "requests": series})

ts["hour"] = ts["datetime"].dt.hour
ts["dayofweek"] = ts["datetime"].dt.dayofweek
ts["dayofyear"] = ts["datetime"].dt.dayofyear

X = ts[["hour", "dayofweek", "dayofyear"]]
y = ts["requests"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

future_dates = pd.date_range(start="2026-01-01", end="2026-12-31 23:00", freq="H")
future = pd.DataFrame({"datetime": future_dates})
future["hour"] = future["datetime"].dt.hour
future["dayofweek"] = future["datetime"].dt.dayofweek
future["dayofyear"] = future["datetime"].dt.dayofyear
future["forecast"] = model.predict(future[["hour","dayofweek","dayofyear"]])

daily = future.resample("D", on="datetime")["forecast"].sum()
